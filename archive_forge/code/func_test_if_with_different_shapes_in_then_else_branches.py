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
def test_if_with_different_shapes_in_then_else_branches(self) -> None:
    then_subgraph = helper.make_graph([make_node('Add', ['current_value', 'add_value'], ['then_output'])], 'then_subgraph', [], [make_tensor_value_info('then_output', TensorProto.UNDEFINED, (1,))])
    else_subgraph = helper.make_graph([make_node('Sub', ['current_value', 'sub_value'], ['else_output'])], 'else_subgraph', [], [make_tensor_value_info('else_output', TensorProto.UNDEFINED, (5,))])
    graph = self._make_graph([('cond', TensorProto.BOOL, (1,)), ('current_value', TensorProto.FLOAT, (1,)), ('add_value', TensorProto.FLOAT, (1,)), ('sub_value', TensorProto.FLOAT, (5,))], [make_node('If', ['cond'], ['if_output'], then_branch=then_subgraph, else_branch=else_subgraph)], [])
    self._assert_inferred(graph, [make_tensor_value_info('if_output', TensorProto.FLOAT, (None,))])