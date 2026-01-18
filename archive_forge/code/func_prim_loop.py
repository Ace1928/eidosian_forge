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
@_onnx_symbolic('prim::Loop')
@_beartype.beartype
def prim_loop(g: jit_utils.GraphContext, *inputs, **attrs) -> List[_C.Value]:
    node = g.original_node
    env = g.env
    params_dict = g.params_dict
    operator_export_type = GLOBALS.operator_export_type
    opset_version = GLOBALS.export_onnx_opset_version
    old_blocks = tuple(node.blocks())
    new_op_outputs, new_block_contexts, new_node = jit_utils.add_op_with_blocks(g, 'Loop', *inputs, outputs=node.outputsSize(), n_blocks=len(old_blocks))
    for old_block, new_block_context in zip(old_blocks, new_block_contexts):
        for i, b_in in enumerate(old_block.inputs()):
            if i == 0 and i < len(inputs):
                b_in.setType(inputs[i].type())
            if i > 0 and i + 1 < len(inputs) and (not isinstance(b_in.type(), _C.OptionalType)):
                b_in.setType(inputs[i + 1].type())
        torch._C._jit_pass_onnx_block(old_block, new_block_context.block, operator_export_type, env, False)
    fixed_outputs = torch._C._jit_pass_fixup_onnx_controlflow_node(new_node, opset_version)
    if GLOBALS.onnx_shape_inference:
        torch._C._jit_pass_onnx_node_shape_type_inference(new_node, params_dict, opset_version)
    return fixed_outputs