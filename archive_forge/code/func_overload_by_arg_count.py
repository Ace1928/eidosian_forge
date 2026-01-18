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
def overload_by_arg_count(fn):

    @functools.wraps(fn)
    @_beartype.beartype
    def wrapper(g, *args):
        overloads = fn(g, *args)
        for overload in overloads:
            arg_descriptors = overload._arg_descriptors
            if len(arg_descriptors) == len(args):
                return overload(g, *args)
        return symbolic_helper._unimplemented(f'aten::{fn.__name__}', f'with {len(args)} arguments')
    return wrapper