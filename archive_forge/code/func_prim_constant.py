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
@_onnx_symbolic('prim::Constant')
@_beartype.beartype
def prim_constant(g: jit_utils.GraphContext, *inputs, **attrs):
    node = g.original_node
    if node.mustBeNone():
        return None
    if isinstance(node.output().type(), _C.DeviceObjType):
        return None
    if node.kindOf('value') == 't':
        return g.op('Constant', value_t=symbolic_helper._node_get(node, 'value'))
    if node.kindOf('value') == 's':
        return g.op('Constant', value_s=symbolic_helper._node_get(node, 'value'))
    if node.output().type().isSubtypeOf(_C.ListType.ofInts()) or node.output().type().isSubtypeOf(_C.ListType.ofFloats()):
        return g.op('Constant', value_t=torch.tensor(symbolic_helper._node_get(node, 'value')))
    if node.output().type().isSubtypeOf(_C.ListType.ofStrings()):
        str_constants = [g.op('Constant', value_s=s) for s in symbolic_helper._node_get(node, 'value')]
        return g.op('prim::ListConstruct', *str_constants)
    raise errors.SymbolicValueError(f"Unsupported prim::Constant kind: '{node.kindOf('value')}'. Please send a bug report at {_constants.PYTORCH_GITHUB_ISSUES_URL}.", node.output())