import functools
import json
import os
from typing import Any, Dict, Optional, Tuple
import torch
import triton
import triton.language as tl
from vllm._C import ops
from vllm.logger import init_logger
from vllm.utils import is_hip
def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor, num_tokens_post_padded: torch.Tensor, mul_routed_weight: bool, top_k: int, config: Dict[str, Any]) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']),)
    fused_moe_kernel[grid](A, B, C, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, B.shape[1], B.shape[2], sorted_token_ids.shape[0], topk_ids.numel(), A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1), C.stride(1), C.stride(2), MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k, compute_type=tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16, **config)