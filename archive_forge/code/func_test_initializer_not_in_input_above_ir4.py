import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_initializer_not_in_input_above_ir4(self):
    nodes = [helper.make_node('BatchNormalization', ['X', 'scale', 'B', 'mean', 'var'], ['Y'])]
    scale_value = [0.55, 0.72]
    scale_tensor = onnx.helper.make_tensor('scale', onnx.TensorProto.FLOAT, [2], scale_value)
    b_value = [0.6, 0.54]
    b_tensor = onnx.helper.make_tensor('B', onnx.TensorProto.FLOAT, [2], b_value)
    mean_value = [0.42, 0.65]
    mean_tensor = onnx.helper.make_tensor('mean', onnx.TensorProto.FLOAT, [2], mean_value)
    var_value = [0.44, 0.89]
    var_tensor = onnx.helper.make_tensor('var', onnx.TensorProto.FLOAT, [2], var_value)
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (1, 2, 2, 3))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (1, 2, 2, 3))], [scale_tensor, b_tensor, mean_tensor, var_tensor])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 11), 12)
    assert converted_model.graph.node[0].op_type == 'BatchNormalization'
    assert converted_model.opset_import[0].version == 12