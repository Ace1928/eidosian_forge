import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_constant_8_9(self) -> None:
    from_opset = 8
    to_opset = 9
    data_type = TensorProto.FLOAT
    output_shape = [2, 3, 4]
    output_value = np.arange(24)
    nodes = [helper.make_node('Constant', inputs=[], outputs=['Y'], value=helper.make_tensor('', data_type, output_shape, output_value))]
    graph = helper.make_graph(nodes, 'test_constant', [], [onnx.helper.make_tensor_value_info('Y', data_type, output_shape)])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert converted_model.graph.node[0].op_type == 'Constant'
    assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
    assert converted_model.opset_import[0].version == to_opset