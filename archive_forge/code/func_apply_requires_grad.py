import os
import torch
from torch.testing import make_tensor  # noqa: F401
from torch.testing._internal.opinfo.core import (  # noqa: F401
def apply_requires_grad(x):
    if not isinstance(x, torch.Tensor) or x.requires_grad or (not requires_grad) or (not (x.is_floating_point() or x.is_complex())):
        return x
    return x.detach().clone().requires_grad_(requires_grad)