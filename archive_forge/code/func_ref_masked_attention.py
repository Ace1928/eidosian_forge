from typing import List, Optional
import importlib
import torch
import torch.nn as nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
from vllm._C import ops
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.triton_kernel.prefix_prefill import (
from vllm.utils import is_hip
def ref_masked_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    query = query.view(-1, self.num_heads, self.head_size)
    key = key.view(-1, self.num_kv_heads, self.head_size)
    value = value.view(-1, self.num_kv_heads, self.head_size)
    seq_len, _, _ = query.shape
    attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=query.dtype, device=query.device), diagonal=1)
    attn_mask = attn_mask * torch.finfo(query.dtype).min
    attn_weights = self.scale * torch.einsum('qhd,khd->hqk', query, key).float()
    attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum('hqk,khd->qhd', attn_weights, value)
    return out