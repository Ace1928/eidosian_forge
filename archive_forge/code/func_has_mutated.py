from collections import namedtuple
from copy import deepcopy
from itertools import combinations
import torch
from torch.fx.operator_schemas import normalize_function
from torch.testing._internal.jit_utils import clone_inputs
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
def has_mutated(before, after, md):
    are_tensors = type(before) == torch.Tensor and type(after) == torch.Tensor
    if are_tensors and before.layout != torch.sparse_csr and (after.layout != torch.sparse_csr):
        return not (before.size() == after.size() and bitwise_equal(before, after) and (md[0] == after.stride()) and (md[1] == after._typed_storage()._cdata))
    return False