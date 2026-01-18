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
@_onnx_symbolic('prim::device')
@_beartype.beartype
def prim_device(g: jit_utils.GraphContext, *inputs, **kwargs) -> None:
    output_type = g.original_node.output().type()
    if isinstance(output_type, _C.DeviceObjType):
        return None
    return symbolic_helper._unimplemented('prim::device', f"output type should be 'DeviceObjType', not '{output_type.kind()}'", g.original_node.output())