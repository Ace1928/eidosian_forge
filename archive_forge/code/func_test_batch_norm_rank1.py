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
def test_batch_norm_rank1(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (128,)), ('scale', TensorProto.FLOAT, (1,)), ('b', TensorProto.FLOAT, (1,)), ('mean', TensorProto.FLOAT, (1,)), ('var', TensorProto.FLOAT, (1,))], [make_node('BatchNormalization', ['x', 'scale', 'b', 'mean', 'var'], ['out'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (128,))])