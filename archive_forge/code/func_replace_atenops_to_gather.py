import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
import onnx
from onnx import ModelProto
from ..utils import logging
from .transformations_utils import (
def replace_atenops_to_gather(model: ModelProto) -> ModelProto:
    """
    Replaces broken ATenOp nodes back to Gather nodes.

    Args:
        model (`onnx.ModelProto`):
            The ONNX model to fix.

    Returns:
        `onnx.ModelProto`: The ONNX model fixed.
    """
    nodes = model.graph.node
    for node in nodes:
        if node.op_type in ['ATenOp', 'ATen']:
            op_num = node.name.split('_')[-1]
            new_node = onnx.helper.make_node('Gather', name='Gather_' + op_num, inputs=[node.input[0], node.input[1]], outputs=node.output)
            model.graph.node.remove(node)
            model.graph.node.insert(int(op_num), new_node)
    onnx.checker.check_model(model)
    return model