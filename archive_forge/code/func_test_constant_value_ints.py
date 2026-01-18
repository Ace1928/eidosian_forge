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
def test_constant_value_ints(self) -> None:
    value_ints = [1, 2, 3]
    graph = self._make_graph([], [make_node('Constant', [], ['y'], value_ints=value_ints)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, [len(value_ints)])])