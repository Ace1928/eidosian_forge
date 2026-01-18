from __future__ import annotations
import dataclasses
import re
import typing
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, registration
@_beartype.beartype
def add_op_with_blocks(graph_context: GraphContext, opname: str, *inputs: _C.Value, outputs: int=1, n_blocks: int=1, **attributes) -> Tuple[Any, Tuple[GraphContext, ...], _C.Node]:
    """Creates an ONNX operator "opname", taking inputs and attributes.

    Args:
        graph_context: The context for the current graph.
        opname: The ONNX operator name, e.g., `Abs` or `Add`, or an operator qualified
            with a namespace, e.g., `aten::add`.
        inputs: The inputs to the operator.
        outputs: The number of outputs this operator returns.
            By default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Value`, representing each output of the ONNX operator
            in order.
        n_blocks: The number of sub-blocks to create in the node.
        attributes: The attributes of the ONNX operator.

    Returns:
        A tuple of (output_values, new_contexts, node) where:
            output_values: One or more output value of this operator
                (see the `outputs` keyword argument for multi-return nodes).
            new_contexts: A tuple of new graph contexts for each sub-block.
            node: The node representing the operator.
    """
    output_values = graph_context.op(opname, *inputs, outputs=outputs, **attributes)
    if isinstance(output_values, Sequence):
        node = output_values[0].node()
    else:
        node = output_values.node()
    new_contexts = []
    for _ in range(n_blocks):
        new_block = node.addBlock()
        new_context = dataclasses.replace(graph_context, block=new_block)
        new_contexts.append(new_context)
    return (output_values, tuple(new_contexts), node)