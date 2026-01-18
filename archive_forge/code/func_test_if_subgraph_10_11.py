import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
def test_if_subgraph_10_11(self) -> None:
    from_opset = 10
    to_opset = 11
    data_type = TensorProto.FLOAT
    data_shape = [2]
    subg1_node = [onnx.helper.make_node('Clip', inputs=['sub_in'], outputs=['sub_out'], min=2.0, max=3.0)]
    subg1_input = [onnx.helper.make_tensor_value_info('sub_in', data_type, data_shape)]
    subg1_output = [onnx.helper.make_tensor_value_info('sub_out', data_type, data_shape)]
    subg1 = helper.make_graph(subg1_node, 'then_g', subg1_input, subg1_output)
    subg2_node = [onnx.helper.make_node('Clip', inputs=['sub_in'], outputs=['sub_out'], min=2.0, max=3.0)]
    subg2_input = [onnx.helper.make_tensor_value_info('sub_in', data_type, data_shape)]
    subg2_output = [onnx.helper.make_tensor_value_info('sub_out', data_type, data_shape)]
    subg2 = helper.make_graph(subg2_node, 'then_g', subg2_input, subg2_output)
    node = [onnx.helper.make_node('If', inputs=['cond'], outputs=['out'], then_branch=subg1, else_branch=subg2)]
    input = [onnx.helper.make_tensor_value_info('cond', TensorProto.BOOL, [])]
    output = [onnx.helper.make_tensor_value_info('out', data_type, data_shape)]
    init = [helper.make_tensor('sub_in', data_type, data_shape, [4.0, 5.0])]
    graph = helper.make_graph(node, 'test_subgraphs', input, output, init)
    converted = self._converted(graph, helper.make_operatorsetid('', from_opset), to_opset)
    assert converted.graph.node[0].op_type == 'If'
    assert converted.opset_import[0].version == to_opset
    assert converted.graph.node[0].attribute[0].g.node[2].op_type == 'Clip'
    assert len(converted.graph.node[0].attribute[0].g.node[2].attribute) == 0
    assert converted.graph.node[0].attribute[1].g.node[2].op_type == 'Clip'
    assert len(converted.graph.node[0].attribute[1].g.node[2].attribute) == 0