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
def test_if_no_shape_in_then_branch(self) -> None:
    then_graph = parse_graph('then_graph () => (then_output) { then_output = ReduceSum <keepdims=0> (X, axes) }')
    else_graph = parse_graph('else_graph () => (else_output) { else_output = ReduceSum <keepdims=0> (X) }')
    graph = self._make_graph([('cond', TensorProto.BOOL, (1,)), ('X', TensorProto.FLOAT, (4, 8, 16)), ('axes', TensorProto.INT64, (1,))], [make_node('If', ['cond'], ['if_output'], then_branch=then_graph, else_branch=else_graph)], [])
    self._assert_inferred(graph, [make_tensor_value_info('if_output', TensorProto.FLOAT, None)])