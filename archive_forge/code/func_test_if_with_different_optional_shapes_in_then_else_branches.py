from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
def test_if_with_different_optional_shapes_in_then_else_branches(self) -> None:
    then_tensor_proto = helper.make_tensor_type_proto(elem_type=TensorProto.UNDEFINED, shape=[1])
    then_optional_type_proto = helper.make_optional_type_proto(then_tensor_proto)
    then_optional_vi = helper.make_value_info('then_optional_output', then_optional_type_proto)
    then_subgraph = helper.make_graph([make_node('Optional', ['then_tensor_value'], ['then_optional_output'])], 'then_subgraph', [], [then_optional_vi])
    else_tensor_proto = helper.make_tensor_type_proto(elem_type=TensorProto.UNDEFINED, shape=[5])
    else_optional_type_proto = helper.make_optional_type_proto(else_tensor_proto)
    else_optional_vi = helper.make_value_info('else_optional_output', else_optional_type_proto)
    else_subgraph = helper.make_graph([make_node('Optional', ['else_tensor_value'], ['else_optional_output'])], 'else_subgraph', [], [else_optional_vi])
    graph = self._make_graph([('cond', TensorProto.BOOL, (1,)), ('then_tensor_value', TensorProto.FLOAT, (1,)), ('else_tensor_value', TensorProto.FLOAT, (5,))], [make_node('If', ['cond'], ['if_output'], then_branch=then_subgraph, else_branch=else_subgraph)], [])
    output_tensor_proto = helper.make_tensor_type_proto(elem_type=TensorProto.FLOAT, shape=(None,))
    output_optional_type_proto = helper.make_optional_type_proto(output_tensor_proto)
    output_optional_vi = helper.make_value_info('if_output', output_optional_type_proto)
    self._assert_inferred(graph, [output_optional_vi])