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
def partial_apply_nontensors(fn, args, kwargs):
    inputs = SplitInputs(args, kwargs)

    def new_fn(*tensors_):
        tensors = iter(tensors_)
        full_args = [args[i] if s == 's' else next(tensors) for i, s in enumerate(inputs.arg_types)]
        full_kwargs = {k: kwargs[k] if s == 's' else next(tensors) for k, s in inputs.kwarg_types.items()}
        return fn(*full_args, **full_kwargs)
    return (new_fn, inputs)