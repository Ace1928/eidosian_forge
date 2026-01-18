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
def fused_moe(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, gating_output: torch.Tensor, topk: int, renormalize: bool, inplace: bool=False, override_config: Optional[Dict[str, Any]]=None) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of weights, w1 and w2, and top-k gating mechanism.
    
    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place. Defaults to False.
    - override_config (Optional[Dict[str, Any]]): Optional override for the kernel configuration.
    
    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    assert hidden_states.shape[0] == gating_output.shape[0], 'Number of tokens mismatch'
    assert hidden_states.shape[1] == w1.shape[2], 'Hidden size mismatch'
    assert gating_output.shape[1] == w1.shape[0], 'Number of experts mismatch'
    assert hidden_states.is_contiguous(), 'Hidden_states must be contiguous'
    assert w1.is_contiguous(), 'Expert weights1 must be contiguous'
    assert w2.is_contiguous(), 'Expert weights2 must be contiguous'
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    M, _ = hidden_states.shape
    E, N, _ = w1.shape
    if is_hip():
        routing_weights = torch.softmax(gating_output, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(routing_weights, topk, dim=-1)
    else:
        import vllm._moe_C as moe_kernels
        topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
        topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        token_expert_indicies = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
        moe_kernels.topk_softmax(topk_weights, topk_ids, token_expert_indicies, gating_output.float())
        del token_expert_indicies
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if override_config:
        config = override_config
    else:
        configs = get_moe_configs(E, w2.shape[2])
        if configs:
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}
            if M <= E:
                config = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}
    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N), device=hidden_states.device, dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2), device=hidden_states.device, dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]), device=hidden_states.device, dtype=hidden_states.dtype)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], E)
    invoke_fused_moe_kernel(hidden_states, w1, intermediate_cache1, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, False, topk_ids.shape[1], config)
    ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))
    invoke_fused_moe_kernel(intermediate_cache2, w2, intermediate_cache3, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, True, 1, config)
    if inplace:
        return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape), dim=1, out=hidden_states)
    return torch.sum(intermediate_cache3.view(*intermediate_cache3.shape), dim=1)