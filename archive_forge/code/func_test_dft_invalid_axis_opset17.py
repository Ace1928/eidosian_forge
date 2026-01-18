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
@parameterized.expand([('last', 3), ('last_negative', -1), ('out_of_range', 4), ('out_of_range_negative', -5)])
def test_dft_invalid_axis_opset17(self, _: str, axis: int) -> None:
    graph = self._make_graph([], [make_node('Constant', [], ['input'], value=make_tensor('input', TensorProto.FLOAT, (2, 5, 5, 2), np.ones((2, 5, 5, 2), dtype=np.float32).flatten())), make_node('DFT', ['input', ''], ['output'], onesided=1, axis=axis)], [])
    with self.assertRaises(onnx.shape_inference.InferenceError):
        self._assert_inferred(graph, [make_tensor_value_info('input', TensorProto.FLOAT, (2, 5, 5, 2)), make_tensor_value_info('output', TensorProto.FLOAT, (2, 3, 5, 2))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 17)])