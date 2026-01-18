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
def test_einsum_incorrect_num_inputs(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (2, 3)), ('y', TensorProto.FLOAT, (2, 3)), ('z', TensorProto.FLOAT, (2, 3))], [make_node('Einsum', ['x', 'y'], ['z'], equation='i,...j, k, l-> i')], [])
    self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)