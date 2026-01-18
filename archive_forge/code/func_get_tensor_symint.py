from typing import Tuple
import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._prims_common import is_expandable_to
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *  # noqa: F403
def get_tensor_symint(tensor, *, coeff=1):
    global _tensor_id_counter
    if tensor not in _tensor_symint_registry:
        _tensor_symint_registry[tensor] = torch._C._get_singleton_int(_tensor_id_counter, coeff)
        _tensor_id_counter += 1
    return _tensor_symint_registry[tensor]