from __future__ import annotations
import builtins
import functools
import math
import sys
import warnings
from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch import _C
from torch.onnx import _constants, _deprecation, _type_utils, errors, symbolic_helper
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
from torch.types import Number
@_beartype.beartype
def reform_weights(g, w, n, intervals):
    slices = [symbolic_helper._slice_helper(g, w, axes=[0], starts=[x * n], ends=[y * n]) for x, y in intervals]
    return g.op('Concat', *slices, axis_i=0)