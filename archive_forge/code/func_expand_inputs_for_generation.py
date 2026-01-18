from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PretrainedConfig
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_idefics import IdeficsConfig
from .perceiver import IdeficsPerceiverResampler
from .vision import IdeficsVisionTransformer
def expand_inputs_for_generation(input_ids, expand_size=1, is_encoder_decoder=False, attention_mask=None, encoder_outputs=None, **model_kwargs):
    expanded_return_idx = torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
    input_ids = input_ids.index_select(0, expanded_return_idx)
    model_kwargs['pixel_values'] = model_kwargs.get('pixel_values', None)
    model_kwargs['image_encoder_embeddings'] = model_kwargs.get('image_encoder_embeddings', None)
    model_kwargs['perceiver_embeddings'] = model_kwargs.get('perceiver_embeddings', None)
    model_kwargs['image_attention_mask'] = model_kwargs.get('image_attention_mask', None)
    if 'token_type_ids' in model_kwargs:
        token_type_ids = model_kwargs['token_type_ids']
        model_kwargs['token_type_ids'] = token_type_ids.index_select(0, expanded_return_idx)
    if attention_mask is not None:
        model_kwargs['attention_mask'] = attention_mask.index_select(0, expanded_return_idx)
    if model_kwargs['image_attention_mask'] is not None:
        model_kwargs['image_attention_mask'] = model_kwargs['image_attention_mask'].index_select(0, expanded_return_idx)
    if model_kwargs['pixel_values'] is not None:
        model_kwargs['pixel_values'] = model_kwargs['pixel_values'].index_select(0, expanded_return_idx)
    elif model_kwargs['image_encoder_embeddings'] is not None:
        model_kwargs['image_encoder_embeddings'] = model_kwargs['image_encoder_embeddings'].index_select(0, expanded_return_idx)
    elif model_kwargs['perceiver_embeddings'] is not None:
        model_kwargs['perceiver_embeddings'] = model_kwargs['perceiver_embeddings'].index_select(0, expanded_return_idx)
    return (input_ids, model_kwargs)