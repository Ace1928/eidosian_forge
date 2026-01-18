from __future__ import annotations
import functools
import sys
import warnings
from typing import Optional, Sequence
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx import (
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration
@_onnx_symbolic('aten::pixel_shuffle')
@symbolic_helper.parse_args('v', 'i')
@_beartype.beartype
def pixel_shuffle(g: jit_utils.GraphContext, self, upscale_factor):
    rank = symbolic_helper._get_tensor_rank(self)
    if rank is not None and rank != 4:
        return symbolic_helper._unimplemented('pixel_shuffle', 'only support 4d input')
    return g.op('DepthToSpace', self, blocksize_i=upscale_factor, mode_s='CRD')