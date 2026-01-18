from typing import List, Tuple
import torch
from torch._vmap_internals import _vmap
from . import forward_ad as fwAD
def jac_func(*inp):
    if outer_jacobian_strategy == 'forward-mode':
        inp = tuple((t.requires_grad_(True) for t in inp))
    jac = jacobian(ensure_single_output_function, inp, create_graph=True)
    _check_requires_grad(jac, 'jacobian', strict=strict)
    return jac