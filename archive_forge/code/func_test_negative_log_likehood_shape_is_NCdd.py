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
def test_negative_log_likehood_shape_is_NCdd(self) -> None:
    N, C = (3, 4)
    graph = self._make_graph([('input', TensorProto.FLOAT, (N, C)), ('target', TensorProto.INT64, (N,))], [make_node('NegativeLogLikelihoodLoss', ['input', 'target'], ['loss'], reduction='none')], [])
    self._assert_inferred(graph, [make_tensor_value_info('loss', TensorProto.FLOAT, (N,))])