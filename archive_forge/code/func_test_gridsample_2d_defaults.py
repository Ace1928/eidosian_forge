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
def test_gridsample_2d_defaults(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, ('N', 'C', 'H', 'W')), ('grid', TensorProto.FLOAT, ('N', 'H_out', 'W_out', 2))], [make_node('GridSample', ['x', 'grid'], ['y'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, ('N', 'C', 'H_out', 'W_out'))])