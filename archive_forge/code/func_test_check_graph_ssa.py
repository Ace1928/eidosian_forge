import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_graph_ssa(self) -> None:
    relu1 = helper.make_node('Relu', ['X'], ['Z'], name='relu1')
    relu2 = helper.make_node('Relu', ['Y'], ['Z'], name='relu2')
    graph = helper.make_graph([relu1, relu2], 'test', inputs=[helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2]), helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])], outputs=[helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 2])])
    self.assertRaises(checker.ValidationError, checker.check_graph, graph)