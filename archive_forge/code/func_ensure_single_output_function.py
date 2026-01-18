from typing import List, Tuple
import torch
from torch._vmap_internals import _vmap
from . import forward_ad as fwAD
def ensure_single_output_function(*inp):
    out = func(*inp)
    is_out_tuple, t_out = _as_tuple(out, 'outputs of the user-provided function', 'hessian')
    _check_requires_grad(t_out, 'outputs', strict=strict)
    if is_out_tuple or not isinstance(out, torch.Tensor):
        raise RuntimeError('The function given to hessian should return a single Tensor')
    if out.nelement() != 1:
        raise RuntimeError('The Tensor returned by the function given to hessian should contain a single element')
    return out.squeeze()