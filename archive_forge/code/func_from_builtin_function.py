from __future__ import annotations
import dataclasses
import types
from typing import Optional, TYPE_CHECKING, Union
import torch._ops
from torch.onnx._internal import _beartype
@classmethod
@_beartype.beartype
def from_builtin_function(cls, builtin_function: types.BuiltinFunctionType) -> OpName:
    """From a builtin function, e.g. operator.add, math.ceil, etc, get the OpName.

        FX graph uses built-in functions to caculate sympy expression. This function
        is used to get the OpName from a builtin function.

        Args:
            builtin_function (types.BuiltinFunctionType): operator.add, math.ceil, etc.

        Returns:
            OpName: _description_
        """
    op = builtin_function.__name__
    module = builtin_function.__module__
    return cls.from_qualified_name(module + '::' + op)