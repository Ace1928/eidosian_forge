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
def test_constantofshape_without_input_shape_scalar(self) -> None:
    graph = self._make_graph([('shape', TensorProto.INT64, (0,))], [make_node('ConstantOfShape', ['shape'], ['y'], value=make_tensor('value', TensorProto.UINT8, (1,), (2,)))], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.UINT8, ())])