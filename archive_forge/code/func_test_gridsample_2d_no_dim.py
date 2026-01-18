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
def test_gridsample_2d_no_dim(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, ('N', 'C', None, None)), ('grid', TensorProto.FLOAT, ('N', None, None, 2))], [make_node('GridSample', ['x', 'grid'], ['y'], mode='linear', padding_mode='border')], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, ('N', 'C', None, None))])