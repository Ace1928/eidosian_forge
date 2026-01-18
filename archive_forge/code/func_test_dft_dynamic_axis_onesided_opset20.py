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
@parameterized.expand([('real', (2, 5, 5, 1)), ('complex', (2, 5, 5, 2))])
def test_dft_dynamic_axis_onesided_opset20(self, _: str, shape: tuple[int, ...]) -> None:
    graph = self._make_graph([('axis', TensorProto.INT64, ())], [make_node('Constant', [], ['input'], value=make_tensor('input', TensorProto.FLOAT, shape, np.ones(shape, dtype=np.float32).flatten())), make_node('DFT', ['input', '', 'axis'], ['output'], onesided=1)], [])
    self._assert_inferred(graph, [make_tensor_value_info('input', TensorProto.FLOAT, shape), make_tensor_value_info('output', TensorProto.FLOAT, (None, None, None, 2))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 20)])