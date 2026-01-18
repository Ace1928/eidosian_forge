import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_flatten_9_8(self) -> None:
    from_opset = 9
    to_opset = 8
    data_type = TensorProto.UINT64
    nodes = [onnx.helper.make_node('Flatten', inputs=['X'], outputs=['Y'], axis=1)]
    graph = helper.make_graph(nodes, 'test_flatten', [onnx.helper.make_tensor_value_info('X', data_type, [2, 3, 4])], [onnx.helper.make_tensor_value_info('Y', data_type, [2, 12])])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert converted_model.graph.node[1].op_type == 'Flatten'
    assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
    assert converted_model.opset_import[0].version == to_opset