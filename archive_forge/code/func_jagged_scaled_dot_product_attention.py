import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn
import torch.nn.functional as F
from torch.backends.cuda import (
from .nested_tensor import NestedTensor
def jagged_scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor]=None, dropout_p=0.0, is_causal=False, scale=None):
    _validate_sdpa_input(query, key, value, attn_mask, dropout_p, is_causal, scale)
    assert isinstance(query, NestedTensor) and isinstance(key, NestedTensor) and isinstance(value, NestedTensor)
    if query.dim() > 3 and key.dim() > 3 and (value.dim() > 3) and (query._ragged_idx == 1):
        from torch.nested._internal.ops import extract_kwargs
        output = F.scaled_dot_product_attention(query._values, key._values, value._values, attn_mask=attn_mask._values if isinstance(attn_mask, NestedTensor) else attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        return NestedTensor(output, **extract_kwargs(query))
    compute_logsumexp = query.requires_grad or key.requires_grad or value.requires_grad
    backend_choice = _select_sdp_backend(query, key, value, attn_mask, dropout_p, is_causal)
    if backend_choice == SDPBackend.FLASH_ATTENTION:
        og_size = query.size(-1)
        query_padded = _pad_last_dim(query, 8, False)
        key_padded = _pad_last_dim(key, 8, False)
        value_padded = _pad_last_dim(value, 8, False)
        og_scale = _calculate_scale(query, scale)
        query_buffer_reshaped, key_buffer_reshaped, value_buffer_reshaped, cumulative_sequence_length_q, cumulative_sequence_length_kv, max_seqlen_batch_q, max_seqlen_batch_kv, output_nt_info = _sdpa_nested_preprocessing(query_padded, key_padded, value_padded)
        attention, logsumexp, philox_seed, philox_offset, debug_attn_mask = torch.ops.aten._flash_attention_forward(query_buffer_reshaped, key_buffer_reshaped, value_buffer_reshaped, cumulative_sequence_length_q, cumulative_sequence_length_kv, max_seqlen_batch_q, max_seqlen_batch_kv, dropout_p, is_causal, False, scale=og_scale)
        attention = NestedTensor(attention, **output_nt_info).transpose(1, 2)
        return _post_process_flash_output(attention, og_size)
    elif backend_choice == SDPBackend.EFFICIENT_ATTENTION:
        query_reshaped, key_reshaped, value_reshaped, cumulative_sequence_length_q, cumulative_sequence_length_kv, max_seqlen_batch_q, _, output_nt_info = _sdpa_nested_preprocessing(query, key, value)
        attention, log_sumexp, seed, offset, max_seqlen_q, max_seqlen_batch_kv = torch.ops.aten._efficient_attention_forward(query_reshaped.unsqueeze(0), key_reshaped.unsqueeze(0), value_reshaped.unsqueeze(0), None, cumulative_sequence_length_q, cumulative_sequence_length_kv, max_seqlen_batch_q, dropout_p, int(is_causal), compute_logsumexp, scale=scale)
        return NestedTensor(attention.squeeze(0), **output_nt_info).transpose(1, 2)
    elif backend_choice == SDPBackend.MATH:
        return torch._scaled_dot_product_attention_math(query, key, value, attn_mask, dropout_p, is_causal, scale=scale)[0]
    else:
        raise RuntimeError('No viable backend for scaled_dot_product_attention was found.')