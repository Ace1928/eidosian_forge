import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_graph_optional_input(self) -> None:
    node = helper.make_node('GivenTensorFill', [''], ['Y'], name='test')
    graph = helper.make_graph([node], 'test', [], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])])
    checker.check_graph(graph)