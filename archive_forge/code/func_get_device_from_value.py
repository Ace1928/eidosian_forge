from __future__ import annotations
import dataclasses
import re
import typing
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, registration
@_beartype.beartype
def get_device_from_value(value: _C.Value) -> Optional[torch.device]:
    if not _is_tensor(value):
        return None
    tensor_type = typing.cast(_C.TensorType, value.type())
    return tensor_type.device()