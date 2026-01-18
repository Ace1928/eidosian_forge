import torch
from torch import Tensor
import itertools
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils import _pytree as pytree
from functools import partial
from torch.utils._mode_utils import no_dispatch, all_same_mode
import torch.autograd.forward_ad as fwAD
from typing import Callable
import re
def maybe_make_dual(dual):
    primal, tangent = dual
    if isinstance(primal, torch.Tensor) and primal.requires_grad:
        return fwAD.make_dual(primal.detach(), tangent)
    elif is_tensorlist(primal):
        return tuple((fwAD.make_dual(pri.detach(), tang) if tang is not None else pri for pri, tang in zip(primal, tangent)))
    return primal