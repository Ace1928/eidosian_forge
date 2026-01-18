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
def gather_leaf_tensors(args, kwargs):
    leaf_tensors = []
    args, args_spec = tree_flatten(args)
    kwargs, kwargs_spec = tree_flatten(kwargs)
    args = args + kwargs
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            continue
        if arg.requires_grad:
            leaf_tensors.append(arg)
    return leaf_tensors