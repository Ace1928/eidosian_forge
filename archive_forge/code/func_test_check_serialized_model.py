import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_serialized_model(self) -> None:
    node = helper.make_node('Relu', ['X'], ['Y'], name='test')
    graph = helper.make_graph([node], 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 2])])
    model = helper.make_model(graph, producer_name='test')
    checker.check_model(model.SerializeToString())