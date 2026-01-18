from collections import namedtuple
from copy import deepcopy
from itertools import combinations
import torch
from torch.fx.operator_schemas import normalize_function
from torch.testing._internal.jit_utils import clone_inputs
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
def bitwise_equal(lhs, rhs):
    if lhs.is_quantized:
        return torch.equal(lhs, rhs)
    else:
        return torch.allclose(lhs, rhs, equal_nan=True)