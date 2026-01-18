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
@classmethod
def from_tensor_var(cls, tx, tensor_var, class_type, torch_function_fn):
    import torch
    kwargs = dict(tensor_var.__dict__)
    assert kwargs.pop('class_type') is torch.Tensor, 'invalid class type in TensorWithTFOverrideVariable.from_tensor_var'
    var = cls(torch_function_fn=torch_function_fn, class_type=class_type, **kwargs)
    var.install_global(tx)
    return var