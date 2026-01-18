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
def test_average_pool_auto_pads(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (30, 4, 7, 6, 4))], [make_node('AveragePool', ['x'], 'z', auto_pad='SAME_UPPER', kernel_shape=[4, 3, 2], strides=[2, 2, 1])], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, (30, 4, 4, 3, 4))])