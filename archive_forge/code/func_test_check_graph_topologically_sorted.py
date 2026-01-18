import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_graph_topologically_sorted(self) -> None:
    n1 = helper.make_node('Scale', ['X'], ['Y'], scale=2.0, name='n1')
    n2 = helper.make_node('Scale', ['Y'], ['Z'], scale=3.0, name='n2')
    graph = helper.make_graph([n2, n1], 'test', inputs=[helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])], outputs=[helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 2])])
    self.assertRaises(checker.ValidationError, checker.check_graph, graph)