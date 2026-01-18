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
def test_batch_norm_invalid(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (128,)), ('scale', TensorProto.FLOAT, (1, 2)), ('b', TensorProto.FLOAT, (1,)), ('mean', TensorProto.FLOAT, (1,)), ('var', TensorProto.FLOAT, (1,))], [make_node('BatchNormalization', ['x', 'scale', 'b', 'mean', 'var'], ['out'])], [])
    self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)