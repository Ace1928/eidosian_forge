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
def test_gemm_no_bias(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (13, 7)), ('y', TensorProto.FLOAT, (7, 17))], [make_node('Gemm', ['x', 'y'], ['out'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (13, 17))])