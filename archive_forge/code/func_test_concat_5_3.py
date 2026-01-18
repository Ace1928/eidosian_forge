import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_concat_5_3(self) -> None:
    nodes = [helper.make_node('Concat', ['X1', 'X2', 'X3', 'X4', 'X5'], ['Y'], axis=0)]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X1', TensorProto.FLOAT, (1,)), helper.make_tensor_value_info('X2', TensorProto.FLOAT, (1,)), helper.make_tensor_value_info('X3', TensorProto.FLOAT, (1,)), helper.make_tensor_value_info('X4', TensorProto.FLOAT, (1,)), helper.make_tensor_value_info('X5', TensorProto.FLOAT, (1,))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (5,))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 5), 3)
    assert converted_model.graph.node[0].op_type == 'Concat'
    assert converted_model.opset_import[0].version == 3