from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
import torch.nn.functional as F
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
from torch.testing._internal.common_nn import module_tests, new_module_tests
from torch.testing._internal.common_utils import is_iterable_of_tensors
import collections
from copy import deepcopy
from typing import Any, Dict, List, Union
import math  # noqa: F401
from torch import inf
def get_script_args(args):
    formals: List[str] = []
    tensors: List[Union[torch.Tensor, List[torch.Tensor]]] = []
    actuals: List[str] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            name = f'i{len(formals)}'
            formals.append(name)
            actuals.append(name)
            tensors.append(arg)
        elif is_iterable_of_tensors(arg):
            name = f'i{len(formals)}'
            formals.append(name + ': List[torch.Tensor]')
            actuals.append(name)
            tensors.append(list(arg))
        elif isinstance(arg, str):
            actuals.append(f"'{arg}'")
        else:
            actuals.append(str(get_constant(arg)))
    return (formals, tensors, actuals)