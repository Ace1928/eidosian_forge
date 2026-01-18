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
def get_item_dyn(self, tx, arg: VariableTracker):
    from .tensor import SymNodeVariable
    if isinstance(arg, SymNodeVariable):
        index = arg.sym_num
    else:
        index = arg.as_python_constant()
    if isinstance(index, slice):
        return SizeVariable(self.items[index])
    else:
        assert isinstance(index, (int, torch.SymInt))
        return self.items[index]