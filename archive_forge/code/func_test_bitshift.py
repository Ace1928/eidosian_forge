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
def test_bitshift(self) -> None:
    graph = self._make_graph([('x', TensorProto.UINT32, (2, 3, 1)), ('y', TensorProto.UINT32, (2, 3, 1))], [make_node('BitShift', ['x', 'y'], 'z', direction='RIGHT')], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.UINT32, (2, 3, 1))])