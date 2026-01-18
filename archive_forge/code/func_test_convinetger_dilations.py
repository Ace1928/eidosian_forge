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
def test_convinetger_dilations(self) -> None:
    graph = self._make_graph([('x', TensorProto.UINT8, (30, 4, 8, 8, 8)), ('y', TensorProto.INT8, (50, 4, 3, 3, 3)), ('x_zero_point', TensorProto.UINT8, ()), ('y_zero_point', TensorProto.UINT8, ())], [make_node('ConvInteger', ['x', 'y', 'x_zero_point', 'y_zero_point'], 'z', dilations=[1, 2, 3])], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.INT32, (30, 50, 6, 4, 2))])