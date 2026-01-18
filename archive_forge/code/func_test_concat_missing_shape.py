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
@parameterized.expand(all_versions_for('Concat'))
def test_concat_missing_shape(self, *_) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (2, 4, 3)), 'y', ('z', TensorProto.FLOAT, (None, None, None))], [make_node('Concat', ['x', 'y', 'z'], ['out'], axis=0)], [])
    self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)