from __future__ import annotations
import inspect
import logging
import operator
import re
import types
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import onnxscript  # type: ignore[import]
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
import torch
import torch.fx
from torch.onnx import _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from torch.utils import _pytree
def filter_incompatible_and_dtype_convert_kwargs(kwargs):
    """Filter out kwargs that are not supported by onnxscript."""
    filtered = {}
    for key, value in kwargs.items():
        if key in {'layout', 'device', 'requires_grad', 'pin_memory', 'memory_format', 'implicit'}:
            continue
        if key == 'dtype':
            if value is None:
                continue
            else:
                value = int(jit_type_utils.JitScalarType.from_dtype(value).onnx_type())
        filtered[key] = value
    return filtered