import collections
import functools
import inspect
import operator
import types
from typing import Dict, List, Optional
import torch
import torch.fx
from ..._guards import Source
from .. import polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..source import AttrSource, GetItemSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .functions import UserFunctionVariable, UserMethodVariable
def check_and_create_method():
    method = inspect.getattr_static(self.tuple_cls, name, None)
    if isinstance(method, classmethod):
        return UserMethodVariable(method.__func__, variables.UserDefinedClassVariable(self.tuple_cls))
    elif isinstance(method, staticmethod):
        return UserFunctionVariable(method.__func__)
    elif inspect.isfunction(method):
        return UserMethodVariable(method, self)
    else:
        return None