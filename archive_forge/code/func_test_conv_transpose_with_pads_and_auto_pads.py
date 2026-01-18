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
def test_conv_transpose_with_pads_and_auto_pads(self) -> None:
    graph = self._make_graph([('X', TensorProto.FLOAT, (1, 1, 2, 2)), ('W', TensorProto.FLOAT, (1, 1, 3, 3)), ('B', TensorProto.FLOAT, (1,))], [make_node('ConvTranspose', ['X', 'W', 'B'], 'Y', auto_pad='SAME_UPPER', strides=[1, 1], pads=[0, 1, 1, 0])], [])
    self.assertRaises(onnx.shape_inference.InferenceError, onnx.shape_inference.infer_shapes, helper.make_model(graph), strict_mode=True)