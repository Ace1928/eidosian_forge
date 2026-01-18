import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_mean_7_8(self) -> None:
    from_opset = 7
    to_opset = 8
    data_type = TensorProto.FLOAT
    data_shape = (3,)
    nodes = [onnx.helper.make_node('Mean', inputs=['X'], outputs=['Y'])]
    graph = helper.make_graph(nodes, 'test_mean', [onnx.helper.make_tensor_value_info('X', data_type, data_shape)], [onnx.helper.make_tensor_value_info('Y', data_type, data_shape)])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert converted_model.graph.node[0].op_type == 'Mean'
    assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
    assert converted_model.opset_import[0].version == to_opset