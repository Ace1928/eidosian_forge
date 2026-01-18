import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_cast_8_9(self) -> None:
    from_opset = 8
    to_opset = 9
    data_type_from = TensorProto.FLOAT
    data_type_to = TensorProto.UINT32
    nodes = [onnx.helper.make_node('Cast', inputs=['X'], outputs=['Y'], to=TensorProto.UINT32)]
    graph = helper.make_graph(nodes, 'test_cast', [onnx.helper.make_tensor_value_info('X', data_type_from, [2, 3])], [onnx.helper.make_tensor_value_info('Y', data_type_to, [2, 3])])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert converted_model.graph.node[0].op_type == 'Cast'
    assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type_to
    assert converted_model.opset_import[0].version == to_opset