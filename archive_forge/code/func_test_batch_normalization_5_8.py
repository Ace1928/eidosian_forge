import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_batch_normalization_5_8(self) -> None:
    nodes = [helper.make_node('BatchNormalization', ['X', 'scale', 'B', 'mean', 'var'], ['Y'])]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (5,)), helper.make_tensor_value_info('scale', TensorProto.FLOAT, (1,)), helper.make_tensor_value_info('B', TensorProto.FLOAT, (1,)), helper.make_tensor_value_info('mean', TensorProto.FLOAT, (1,)), helper.make_tensor_value_info('var', TensorProto.FLOAT, (1,))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (5,))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 5), 8)
    assert converted_model.graph.node[0].op_type == 'BatchNormalization'
    assert converted_model.opset_import[0].version == 8