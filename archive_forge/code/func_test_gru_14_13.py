import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_gru_14_13(self) -> None:
    from_opset = 14
    to_opset = 13
    data_type = TensorProto.FLOAT
    seq_length = 1
    batch_size = 2
    input_size = 3
    num_directions = 1
    hidden_size = 5
    nodes = [onnx.helper.make_node('GRU', inputs=['X', 'W', 'R'], outputs=['', 'Y_h'], hidden_size=hidden_size, layout=0)]
    graph = helper.make_graph(nodes, 'test_gru', [onnx.helper.make_tensor_value_info('X', data_type, [seq_length, batch_size, input_size]), onnx.helper.make_tensor_value_info('W', data_type, [num_directions, 3 * hidden_size, input_size]), onnx.helper.make_tensor_value_info('R', data_type, [num_directions, 3 * hidden_size, hidden_size]), onnx.helper.make_tensor_value_info('B', data_type, [num_directions, 6 * hidden_size])], [onnx.helper.make_tensor_value_info('Y_h', data_type, [num_directions, batch_size, hidden_size])])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert converted_model.graph.node[0].op_type == 'GRU'
    assert converted_model.opset_import[0].version == to_opset
    assert len(converted_model.graph.node[0].attribute) == 1