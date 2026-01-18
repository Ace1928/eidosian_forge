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
@staticmethod
def list_compare(tx, op, left, right):
    from .builtin import BuiltinVariable
    eq_result = BaseListVariable.list_eq(tx, left, right)
    if op is operator.eq:
        return eq_result
    elif op is operator.ne:
        return BuiltinVariable(operator.not_).call_function(tx, [eq_result], {})
    else:
        unimplemented(f'list_compare {left} {op} {right}')