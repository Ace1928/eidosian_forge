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
def test_constantofshape_with_shape_zero(self) -> None:
    graph = self._make_graph([], [make_node('Constant', [], ['shape'], value=make_tensor('shape', TensorProto.INT64, (1,), (0,))), make_node('ConstantOfShape', ['shape'], ['y'], value=make_tensor('value', TensorProto.INT32, (1,), (2,)))], [])
    self._assert_inferred(graph, [make_tensor_value_info('shape', TensorProto.INT64, (1,)), make_tensor_value_info('y', TensorProto.INT32, (0,))])