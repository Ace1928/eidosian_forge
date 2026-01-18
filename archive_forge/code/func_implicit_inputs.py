from __future__ import annotations
import abc
from typing import Any, ClassVar, Iterable
import numpy as np
from onnx import TensorProto
from onnx.defs import get_all_schemas_with_history, get_schema, onnx_opset_version
from onnx.helper import make_node, make_tensor_type_proto, np_dtype_to_tensor_dtype
from onnx.numpy_helper import to_array, unpack_int4
from onnx.onnx_pb import AttributeProto, GraphProto, NodeProto, TypeProto
from onnx.reference.custom_element_types import (
@staticmethod
def implicit_inputs(graph: GraphProto) -> list[str]:
    """Returns all varibles not registered as inputs and not produced by
        an node inside the graph. This inputs are part of the context
        existing in the graph calling this one.
        """
    if not isinstance(graph, GraphProto):
        raise TypeError(f'Unexpected type {type(graph)!r}.')
    local = set()
    known = set()
    for init in graph.initializer:
        known.add(init.name)
    for sparse_init in graph.sparse_initializer:
        known.add(sparse_init.name)
    for inp in graph.input:
        known.add(inp.name)
    for node in graph.node:
        for o in node.output:
            known.add(o)
        for i in node.input:
            if i not in known:
                local.add(i)
    return list(local)