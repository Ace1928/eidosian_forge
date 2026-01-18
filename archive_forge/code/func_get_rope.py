import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from vllm._C import ops
def get_rope(head_size: int, rotary_dim: int, max_position: int, base: int, is_neox_style: bool=True, rope_scaling: Optional[Dict[str, Any]]=None) -> RotaryEmbedding:
    key = (head_size, rotary_dim, max_position, base, is_neox_style, tuple(rope_scaling.items()) if rope_scaling is not None else None)
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]
    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, is_neox_style)
    else:
        scaling_type = rope_scaling['type']
        scaling_factor = rope_scaling['factor']
        if scaling_type == 'linear':
            rotary_emb = LinearScalingRotaryEmbedding(head_size, rotary_dim, max_position, base, is_neox_style, scaling_factor)
        elif scaling_type == 'dynamic':
            rotary_emb = DynamicNTKScalingRotaryEmbedding(head_size, rotary_dim, max_position, base, is_neox_style, scaling_factor)
        elif scaling_type == 'yarn':
            original_max_position = rope_scaling['original_max_position_embeddings']
            extra_kwargs = {k: v for k, v in rope_scaling.items() if k in ('extrapolation_factor', 'attn_factor', 'beta_fast', 'beta_slow')}
            rotary_emb = YaRNScalingRotaryEmbedding(head_size, rotary_dim, original_max_position, base, is_neox_style, scaling_factor, **extra_kwargs)
        else:
            raise ValueError(f'Unknown RoPE scaling type {scaling_type}')
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb