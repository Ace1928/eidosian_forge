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
def is_tensorlist(lst):
    if not isinstance(lst, list) and (not isinstance(lst, tuple)):
        return False
    if len(lst) == 0:
        return False
    all_tensors = all((isinstance(elt, torch.Tensor) for elt in lst))
    if all_tensors:
        return True
    exists_one_tensor = all((isinstance(elt, torch.Tensor) for elt in lst))
    if exists_one_tensor:
        raise RuntimeError('This test assumes that PyTorch APIs cannot take mixed lists of Tensor and other things')
    return False