from __future__ import annotations
import functools
import sys
import warnings
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.onnx
from torch import _C
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
def is_none_value(value):
    if value is None:
        return True
    return isinstance(value, torch._C.Value) and value.node().kind() == 'prim::Constant' and isinstance(value.type(), _C.NoneType)