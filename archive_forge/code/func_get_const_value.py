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
def get_const_value(list_or_value):
    if isinstance(list_or_value, (list, torch.Tensor)):
        if len(list_or_value) == 1:
            return list_or_value[0]
        return None
    return symbolic_helper._maybe_get_const(list_or_value, 'i')