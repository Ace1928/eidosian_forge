from __future__ import annotations
from typing import (
import numpy
import onnx
import torch
from torch._subclasses import fake_tensor
def from_python_type_to_onnx_attribute_type(dtype: type, is_sequence: bool=False) -> Optional[onnx.defs.OpSchema.AttrType]:
    import onnx.defs
    _PYTHON_TYPE_TO_ONNX_ATTRIBUTE_TYPE = {float: onnx.defs.OpSchema.AttrType.FLOAT, int: onnx.defs.OpSchema.AttrType.INT, str: onnx.defs.OpSchema.AttrType.STRING, bool: onnx.defs.OpSchema.AttrType.INT}
    _SEQUENCE_TYPE_TO_ONNX_ATTRIBUTE_TYPE = {float: onnx.defs.OpSchema.AttrType.FLOATS, int: onnx.defs.OpSchema.AttrType.INTS, str: onnx.defs.OpSchema.AttrType.STRINGS, bool: onnx.defs.OpSchema.AttrType.INTS}
    if is_sequence:
        return _SEQUENCE_TYPE_TO_ONNX_ATTRIBUTE_TYPE.get(dtype)
    return _PYTHON_TYPE_TO_ONNX_ATTRIBUTE_TYPE.get(dtype)