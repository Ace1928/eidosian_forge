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
def nontensors_match(self, other: 'SplitInputs'):
    if self.arg_types != other.arg_types:
        return False
    if self.kwarg_types != other.kwarg_types:
        return False
    if self.kwarg_order != other.kwarg_order:
        return False
    if self.nontensor_args != other.nontensor_args:
        return False
    if self.nontensor_kwargs != other.nontensor_kwargs:
        return False
    return True