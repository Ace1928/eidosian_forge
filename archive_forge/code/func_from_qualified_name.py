from __future__ import annotations
import dataclasses
import types
from typing import Optional, TYPE_CHECKING, Union
import torch._ops
from torch.onnx._internal import _beartype
@classmethod
@_beartype.beartype
def from_qualified_name(cls, qualified_name: str) -> OpName:
    """When the name is <namespace>::<op_name>[.<overload>]"""
    namespace, opname_overload = qualified_name.split('::')
    op_name, *overload = opname_overload.split('.', 1)
    overload = overload[0] if overload else 'default'
    return cls(namespace, op_name, overload)