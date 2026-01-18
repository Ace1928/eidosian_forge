from typing import Optional
import torch
from torch import nn
from .. import _is_triton_available
def rms_norm_add(x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor], eps: float=1e-06):
    """
    An addition fused with rms_norm.

        z = rms_norm_add(x, y, weight, eps)

    is equivalent to

        x += y
        z = rms_norm(x, weight, eps)

    where x, y and z are all contiguous.

    This functionality is experimental. Its API might be changed without warnings.
    Use it at your own risk.
    """
    if torch.is_grad_enabled() and (x.requires_grad or y.requires_grad or (weight is not None and weight.requires_grad)):
        raise ValueError('Gradients not supported.')
    assert _is_triton_available()
    from ._triton.rmsnorm_kernels import _rms_norm_add_forward
    return _rms_norm_add_forward(x, y, weight, eps)