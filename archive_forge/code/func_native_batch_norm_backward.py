import inspect
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch._decomp
from torch import Tensor
from torch._prims_common.wrappers import _maybe_remove_out_wrapper
@register_decomposition_for_jvp(aten.native_batch_norm_backward)
def native_batch_norm_backward(grad_out: Tensor, input: Tensor, weight: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], save_mean: Optional[Tensor], save_invstd: Optional[Tensor], train: bool, eps: float, output_mask: List[bool]) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    input_shape = input.shape
    input_rank = input.dim()
    assert input_rank >= 2, 'rank of the input must be at least 2'
    axis = 1
    num_features = prod(input_shape) / input_shape[axis]
    mean = save_mean
    invstd = save_invstd
    if train:
        assert save_mean is not None and save_invstd is not None, 'when train=True, save_mean and save_invstd are required'
        reduciton_dims = [0] + list(range(2, input.dim()))
        assert invstd is not None
        mean, invstd = recompute_mean_var(input, invstd, reduciton_dims, keepdim=False)
    else:
        assert running_mean is not None and running_var is not None
        mean = running_mean
        invstd = torch.rsqrt(running_var + eps)
    assert invstd is not None and mean is not None
    broadcast_mask = [1] * input_rank
    broadcast_mask[axis] = input_shape[axis]
    reduction_axes: List[int] = []
    for i in range(input_rank):
        if i != axis:
            reduction_axes.append(i)
    mean = torch.reshape(mean, broadcast_mask)
    norm = 1.0 / num_features
    grad_output_sum = torch.sum(grad_out, reduction_axes)
    dot_p = torch.sum(grad_out * (input - mean), reduction_axes)
    grad_mean = torch.reshape(grad_output_sum * norm, broadcast_mask)
    proj_scale = torch.reshape(torch.mul(dot_p * norm, invstd * invstd), broadcast_mask)
    if weight is None:
        grad_scale = torch.reshape(invstd, broadcast_mask) * 1.0
    else:
        grad_scale = torch.reshape(invstd * weight, broadcast_mask)
    if train:
        proj = (input - mean) * proj_scale
        grad_input = (grad_out - proj - grad_mean) * grad_scale
    else:
        grad_input = grad_out * grad_scale
    if output_mask[1]:
        grad_weight = dot_p * invstd
    elif weight is not None:
        grad_weight = torch.zeros_like(weight)
    else:
        grad_weight = torch.zeros(())
    if output_mask[2]:
        grad_bias = grad_output_sum
    else:
        grad_bias = torch.zeros_like(grad_output_sum)
    return (grad_input, grad_weight, grad_bias)