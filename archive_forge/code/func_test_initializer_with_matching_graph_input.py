import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
def test_initializer_with_matching_graph_input(self) -> None:
    add = helper.make_node('Add', ['X', 'Y_Initializer'], ['Z'])
    value_info = [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1])]
    graph = helper.make_graph([add], 'test', [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1]), helper.make_tensor_value_info('Y_Initializer', TensorProto.FLOAT, [1])], [helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1])], [helper.make_tensor('Y_Initializer', TensorProto.FLOAT, [1], [1])], doc_string=None, value_info=value_info)
    graph_str = helper.printable_graph(graph)
    self.assertTrue(') optional inputs with matching initializers (\n  %Y_Initializer[FLOAT, 1]' in graph_str, graph_str)