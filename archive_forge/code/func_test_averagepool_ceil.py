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
def test_averagepool_ceil(self) -> None:
    graph = self._make_graph([('X', TensorProto.FLOAT, (1, 1, 4, 4))], [make_node('AveragePool', ['X'], ['Y'], kernel_shape=[3, 3], strides=[2, 2], ceil_mode=True)], [])
    self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (1, 1, 2, 2))])