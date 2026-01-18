import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_maxpool_down(self) -> None:
    nodes = [helper.make_node('MaxPool', ['X'], ['Y'], kernel_shape=[1, 1])]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (5, 5, 5, 5))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (5, 5, 5, 5))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 8), 1)
    assert converted_model.graph.node[0].op_type == 'MaxPool'
    assert converted_model.opset_import[0].version == 1