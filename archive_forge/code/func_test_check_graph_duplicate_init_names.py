import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_graph_duplicate_init_names(self) -> None:
    node = helper.make_node('Relu', ['X'], ['Y'], name='test')
    graph = helper.make_graph([node], 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])])
    checker.check_graph(graph)
    graph.initializer.extend([self._sample_float_tensor])
    graph.initializer[0].name = 'X'
    sparse = self.make_sparse([100], [13, 17, 19], [3], [9, 27, 81], 'X')
    graph.sparse_initializer.extend([sparse])
    self.assertRaises(checker.ValidationError, checker.check_graph, graph)