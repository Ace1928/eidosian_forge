import contextlib
import functools
import inspect
import itertools
import logging
import math
import operator
import types
from collections import defaultdict, OrderedDict
from typing import Dict, List
import torch
from torch import sym_float, sym_int
from .. import config, polyfill, variables
from ..exc import (
from ..guards import GuardBuilder, install_guard
from ..replay_record import DummyModule
from ..source import AttrSource, GetItemSource, is_constant_source, TypeSource
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .constant import ConstantVariable
from .ctx_manager import EventVariable, StreamVariable
from .dicts import ConstDictVariable, DefaultDictVariable, SetVariable
from .lists import (
from .tensor import FakeItemVariable, SymNodeVariable, UnspecializedPythonVariable
from .user_defined import UserDefinedVariable
def call_isinstance(self, tx, arg, isinstance_type):
    arg_type = arg.python_type()
    isinstance_type = isinstance_type.as_python_constant()
    if isinstance(arg, variables.TensorVariable) and arg.dtype is not None:

        def _tensor_isinstance(tensor_var, tensor_type):

            def check_type(ty):
                if ty not in tensortype_to_dtype:
                    return issubclass(arg.python_type(), ty)
                dtypes = tensortype_to_dtype[ty]
                return arg.dtype in dtypes
            if type(tensor_type) is tuple:
                return any((check_type(ty) for ty in tensor_type))
            else:
                return check_type(tensor_type)
        return variables.ConstantVariable.create(_tensor_isinstance(arg, isinstance_type))
    if isinstance(arg, variables.UserDefinedObjectVariable) and isinstance(arg.value, types.MemberDescriptorType):
        unimplemented(f'isinstance called on UserDefinedClass {arg} {isinstance_type}')
    if isinstance(arg, variables.UserDefinedObjectVariable) and '__instancecheck__' in isinstance_type.__class__.__dict__:
        return variables.ConstantVariable.create(isinstance_type.__class__.__instancecheck__(isinstance_type, arg.value))
    try:
        val = issubclass(arg_type, isinstance_type)
    except TypeError:
        val = arg_type is isinstance_type
    return variables.ConstantVariable.create(val)