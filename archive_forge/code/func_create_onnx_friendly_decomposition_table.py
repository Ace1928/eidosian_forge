from __future__ import annotations
from typing import Callable, Dict, Set, Union
import torch
import torch._ops
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import registration
@_beartype.beartype
def create_onnx_friendly_decomposition_table(registry) -> Dict[torch._ops.OperatorBase, Callable]:
    """
    This function creates a dictionary of op overloads and their decomposition functions
    for ops that do not have ONNX symbolic functions. If an op already has an ONNX symbolic function,
    its decomposition function is excluded from the table. The decomposition table is a subset of PyTorch's
    built-in aten-to-aten decomposition.

    Args:
        registry (torch.onnx.OnnxRegistry): The ONNX registry for PyTorch.

    Returns:
        Dict[torch._ops.OperatorBase, Callable]: A dictionary that maps op overloads to their corresponding
        decomposition functions.
    """
    decomposition_table: Dict[torch._ops.OperatorBase, Callable] = {}
    _ONNX_SUPPORT_OP_OVERLOADS = _create_onnx_supports_op_overload_table(registry)
    for op_overload, decomp_fn in torch._decomp.decomposition_table.items():
        if 'torch._refs' in decomp_fn.__module__ or op_overload in _ONNX_SUPPORT_OP_OVERLOADS:
            continue
        decomposition_table[op_overload] = decomp_fn
    return decomposition_table