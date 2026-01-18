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
def test_affinegrid_3d(self) -> None:
    N, C, D, H, W = (2, 3, 4, 5, 6)
    graph = self._make_graph([('theta', TensorProto.FLOAT, (N, 3, 4)), ('size', TensorProto.INT64, (5,))], [make_node('AffineGrid', ['theta', 'size'], ['grid'])], [], initializer=[make_tensor('size', TensorProto.INT64, (5,), (N, C, D, H, W))])
    self._assert_inferred(graph, [make_tensor_value_info('grid', TensorProto.FLOAT, (N, D, H, W, 3))])