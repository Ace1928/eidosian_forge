import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_sparse_matmul(self) -> None:
    M = 5
    N = 10
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [N])
    sparse_tensor = self.make_sparse([M, N], [2, 3, 1], [3], [3, 11, 37])
    node1 = helper.make_node('Constant', [], ['C'], sparse_value=sparse_tensor)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [M])
    node2 = helper.make_node('MatMul', ['C', 'X'], ['Y'])
    graph = helper.make_graph([node1, node2], 'sparse_matmul', [X], [Y])
    checker.check_graph(graph)