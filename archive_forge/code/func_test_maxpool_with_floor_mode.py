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
def test_maxpool_with_floor_mode(self) -> None:
    graph = self._make_graph([('X', TensorProto.FLOAT, (32, 288, 35, 35))], [make_node('MaxPool', ['X'], ['Y'], kernel_shape=[2, 2], strides=[2, 2], ceil_mode=False)], [])
    self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (32, 288, 17, 17))])