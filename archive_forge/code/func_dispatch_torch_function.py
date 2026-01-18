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
def dispatch_torch_function(tx, fn, args, kwargs):
    """Gathers all args that are TensorWithTFOverrideVariable and dispatches based on the ordering in _get_overloaded_args"""
    all_args = pytree.arg_tree_leaves(*args, **kwargs)
    overloaded_args = _get_overloaded_args([arg for arg in all_args if isinstance(arg, TensorWithTFOverrideVariable)], lambda x: x.class_type)
    for arg in overloaded_args:
        res = arg.call_torch_function(tx, fn, TupleVariable([arg.subclass_type_var() for arg in overloaded_args]), args, kwargs)
        if not (isinstance(res, ConstantVariable) and res.value is NotImplemented):
            return res
    unimplemented(f'All __torch_function__ overrides for call {fn} with args {args} and kwargs {kwargs} returned NotImplemented')