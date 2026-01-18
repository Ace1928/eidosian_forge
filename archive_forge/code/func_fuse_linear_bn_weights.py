from __future__ import annotations
import copy
from typing import Optional, Tuple, TypeVar
import torch
def fuse_linear_bn_weights(linear_w: torch.Tensor, linear_b: Optional[torch.Tensor], bn_rm: torch.Tensor, bn_rv: torch.Tensor, bn_eps: float, bn_w: torch.Tensor, bn_b: torch.Tensor) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Fuse linear module parameters and BatchNorm module parameters into new linear module parameters.

    Args:
        linear_w (torch.Tensor): Linear weight.
        linear_b (Optional[torch.Tensor]): Linear bias.
        bn_rm (torch.Tensor): BatchNorm running mean.
        bn_rv (torch.Tensor): BatchNorm running variance.
        bn_eps (float): BatchNorm epsilon.
        bn_w (torch.Tensor): BatchNorm weight.
        bn_b (torch.Tensor): BatchNorm bias.
        transpose (bool, optional): If True, transpose the conv weight. Defaults to False.

    Returns:
        Tuple[torch.nn.Parameter, torch.nn.Parameter]: Fused linear weight and bias.
    """
    if linear_b is None:
        linear_b = torch.zeros_like(bn_rm)
    bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)
    fused_w = linear_w * bn_scale.unsqueeze(-1)
    fused_b = (linear_b - bn_rm) * bn_scale + bn_b
    return (torch.nn.Parameter(fused_w, linear_w.requires_grad), torch.nn.Parameter(fused_b, linear_b.requires_grad))