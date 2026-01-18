import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
import torch
from onnx.tools import update_model_dims
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.file_utils import add_end_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithPast
import onnxruntime
from ..exporters.onnx import MODEL_TYPES_REQUIRING_POSITION_IDS, main_export
from ..onnx.utils import check_model_uses_external_data
from ..utils import NormalizedConfigManager, check_if_transformers_greater
from ..utils.modeling_utils import MODEL_TO_PATCH_FOR_PAST
from ..utils.save_utils import maybe_save_preprocessors
from .constants import DECODER_MERGED_ONNX_FILE_PATTERN, DECODER_ONNX_FILE_PATTERN, DECODER_WITH_PAST_ONNX_FILE_PATTERN
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .models.bloom import bloom_convert_to_bloom_cache, bloom_convert_to_standard_cache
from .utils import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, ONNX_WEIGHTS_NAME
def prepare_past_key_values(self, input_ids: Union[None, torch.LongTensor, np.ndarray], past_key_values: Union[None, Tuple[torch.FloatTensor], Tuple[np.ndarray]], use_torch: bool):
    sequence_length = input_ids.shape[1]
    constructor = torch if use_torch else np
    if self.use_merged:
        use_cache_branch = constructor.full((1,), past_key_values is not None)
    else:
        use_cache_branch = None
    if use_torch and use_cache_branch is not None:
        use_cache_branch = use_cache_branch.to(self.device)
    pkv_output_shape = {}
    if past_key_values is None:
        batch_size = input_ids.shape[0]
        if self.model_type in {'mistral', 'llama'}:
            num_attention_heads = self.normalized_config.num_key_value_heads
        else:
            num_attention_heads = self.normalized_config.num_attention_heads
        embed_size_per_head = self.normalized_config.hidden_size // self.normalized_config.num_attention_heads
        dtype = constructor.float16 if self.use_fp16 else constructor.float32
        if self.model_type == 'bloom':
            shape_value = (batch_size * num_attention_heads, 0, embed_size_per_head)
            shape_key = (batch_size * num_attention_heads, embed_size_per_head, 0)
            key = constructor.zeros(shape_key, dtype=dtype)
            value = constructor.zeros(shape_value, dtype=dtype)
            if use_torch:
                key = key.to(self.device)
                value = value.to(self.device)
            past_key_values = tuple((key_or_value for _ in range(len(self.key_value_input_names) // 2) for key_or_value in [key, value]))
            for name, value in zip(self.key_value_output_names, past_key_values):
                shape = [*value.shape]
                index = 1 if 'value' in name else 2
                shape[index] += sequence_length
                pkv_output_shape[name] = shape
        elif self.model_type == 'gpt_bigcode':
            shape_key_and_value = (batch_size, 0, embed_size_per_head * 2)
            key_and_value = constructor.zeros(shape_key_and_value, dtype=dtype)
            if use_torch:
                key_and_value = key_and_value.to(self.device)
            past_key_values = tuple((key_and_value for _ in range(len(self.key_value_input_names))))
            for name, value in zip(self.key_value_output_names, past_key_values):
                shape = [*value.shape]
                shape[1] += sequence_length
                pkv_output_shape[name] = shape
        else:
            num_key_value_heads = self.num_key_value_heads if self.model_type == 'falcon' else num_attention_heads
            shape = (batch_size, num_key_value_heads, 0, embed_size_per_head)
            key_or_value = constructor.zeros(shape, dtype=dtype)
            if use_torch:
                key_or_value = key_or_value.to(self.device)
            past_key_values = tuple((key_or_value for _ in range(len(self.key_value_input_names))))
            for name, value in zip(self.key_value_output_names, past_key_values):
                shape = [*value.shape]
                shape[2] += sequence_length
                pkv_output_shape[name] = shape
    return (use_cache_branch, past_key_values, pkv_output_shape)