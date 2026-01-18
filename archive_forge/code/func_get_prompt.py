from __future__ import annotations
import collections
import inspect
import os
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Optional, Union
import packaging.version
import torch
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory
from huggingface_hub import ModelCard, ModelCardData, hf_hub_download
from safetensors.torch import save_file as safe_save_file
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import PushToHubMixin
from . import __version__
from .config import PeftConfig
from .tuners import (
from .utils import (
def get_prompt(self, batch_size: int, task_ids: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
        Returns the virtual prompts to use for Peft. Only applicable when using a prompt learning method.
        """
    peft_config = self.active_peft_config
    prompt_encoder = self.prompt_encoder[self.active_adapter]
    prompt_tokens = self.prompt_tokens[self.active_adapter].unsqueeze(0).expand(batch_size, -1).to(prompt_encoder.embedding.weight.device)
    if peft_config.peft_type == PeftType.PREFIX_TUNING:
        prompt_tokens = prompt_tokens[:, :peft_config.num_virtual_tokens]
        if peft_config.inference_mode:
            past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
        else:
            past_key_values = prompt_encoder(prompt_tokens)
        if self.base_model_torch_dtype is not None:
            past_key_values = past_key_values.to(self.base_model_torch_dtype)
        past_key_values = past_key_values.view(batch_size, peft_config.num_virtual_tokens, peft_config.num_layers * 2, peft_config.num_attention_heads, peft_config.token_dim // peft_config.num_attention_heads)
        if peft_config.num_transformer_submodules == 2:
            past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(peft_config.num_transformer_submodules * 2)
        if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
            post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
            past_key_values = post_process_fn(past_key_values)
        return past_key_values
    else:
        if peft_config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            prompts = prompt_encoder(prompt_tokens, task_ids)
        elif peft_config.inference_mode:
            prompts = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
        else:
            prompts = prompt_encoder(prompt_tokens)
        return prompts