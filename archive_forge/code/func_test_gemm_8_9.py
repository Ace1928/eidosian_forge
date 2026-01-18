import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_gemm_8_9(self) -> None:
    from_opset = 8
    to_opset = 9
    data_type = TensorProto.FLOAT
    nodes = [onnx.helper.make_node('Gemm', inputs=['X1', 'X2', 'X3'], outputs=['Y'])]
    graph = helper.make_graph(nodes, 'test_gemm', [onnx.helper.make_tensor_value_info('X1', data_type, [3, 4]), onnx.helper.make_tensor_value_info('X2', data_type, [4, 3]), onnx.helper.make_tensor_value_info('X3', data_type, [3, 3])], [onnx.helper.make_tensor_value_info('Y', data_type, [3, 3])])
    converted_model = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert converted_model.graph.node[0].op_type == 'Gemm'
    assert converted_model.graph.output[0].type.tensor_type.elem_type == data_type
    assert converted_model.opset_import[0].version == to_opset