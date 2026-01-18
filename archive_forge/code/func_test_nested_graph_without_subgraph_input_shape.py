import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_nested_graph_without_subgraph_input_shape(self) -> None:
    n1 = helper.make_node('Scale', ['X'], ['Y'], scale=2.0, name='n1')
    n2 = helper.make_node('Scale', ['Y'], ['Z'], scale=3.0, name='n2')
    input_x = onnx.ValueInfoProto()
    input_x.name = 'X'
    graph = helper.make_graph([n1, n2], 'nested', inputs=[], outputs=[helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 2])])
    i1 = helper.make_node('If', ['cond'], ['Z'], then_branch=graph, else_branch=graph)
    graph = helper.make_graph([i1], 'test', inputs=[helper.make_tensor_value_info('cond', TensorProto.BOOL, [1]), helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])], outputs=[helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 2])])
    checker.check_graph(graph)