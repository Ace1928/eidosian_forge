import inspect
from typing import Dict, List
import torch.utils._pytree as pytree
from torch.overrides import _get_overloaded_args, get_default_nowrap_functions
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GlobalSource
from ..utils import is_tensor_base_attr_getter
from .base import VariableTracker
from .constant import ConstantVariable
from .lists import TupleVariable
from .tensor import TensorVariable
from .user_defined import UserDefinedClassVariable
def can_dispatch_torch_function(tx, args, kwargs):
    if tx.output.torch_function_enabled:
        all_args = pytree.arg_tree_leaves(*args, **kwargs)
        return any((isinstance(arg, TensorWithTFOverrideVariable) for arg in all_args))
    else:
        return False