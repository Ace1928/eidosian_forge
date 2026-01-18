import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_batchnormalization_9_8(self) -> None:
    from_opset = 9
    to_opset = 8
    data_type = TensorProto.FLOAT
    nodes = [onnx.helper.make_node('BatchNormalization', inputs=['X', 'scale', 'B', 'mean', 'var'], outputs=['Y'])]
    input_shape = (2, 3, 4, 5)
    x = onnx.helper.make_tensor_value_info('X', data_type, input_shape)
    scale = onnx.helper.make_tensor_value_info('scale', data_type, [input_shape[1]])
    B = onnx.helper.make_tensor_value_info('B', data_type, [input_shape[1]])
    mean = onnx.helper.make_tensor_value_info('mean', data_type, [input_shape[1]])
    var = onnx.helper.make_tensor_value_info('var', data_type, [input_shape[1]])
    y = onnx.helper.make_tensor_value_info('Y', data_type, input_shape)
    graph = onnx.helper.make_graph(nodes, 'test_batchnormalization', [x, scale, B, mean, var], [y])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert converted_model.graph.node[0].op_type == 'BatchNormalization'
    assert converted_model.opset_import[0].version == to_opset