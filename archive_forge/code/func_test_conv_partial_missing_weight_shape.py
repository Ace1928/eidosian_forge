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
def test_conv_partial_missing_weight_shape(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (30, 4, 7, 6, 4)), ('y', TensorProto.FLOAT, (50, 4, None, 3, 3))], [make_node('Conv', ['x', 'y'], 'z', pads=[1, 1, 2, 0, 1, 2])], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, None)])