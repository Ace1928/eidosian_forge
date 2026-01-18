import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_axes_input_to_attr_13_12(self) -> None:
    nodes = [helper.make_node('Constant', [], ['axes'], value=helper.make_tensor('', TensorProto.INT64, [1], [0])), helper.make_node('ReduceSum', ['X', 'axes'], ['Y'])]
    graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, (5, 5))], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, (1, 5))])
    converted_model = self._converted(graph, helper.make_operatorsetid('', 13), 12)
    assert converted_model.graph.node[0].op_type == 'ReduceSum'
    assert converted_model.opset_import[0].version == 12