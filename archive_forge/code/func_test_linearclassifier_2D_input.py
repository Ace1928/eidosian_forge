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
@unittest.skipUnless(ONNX_ML, 'ONNX_ML required to test ai.onnx.ml operators')
def test_linearclassifier_2D_input(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (4, 5))], [make_node('LinearClassifier', ['x'], ['y', 'z'], domain=ONNX_ML_DOMAIN, coefficients=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], intercepts=[2.0, 2.0, 3.0], classlabels_ints=[1, 2, 3])], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (4,)), make_tensor_value_info('z', TensorProto.FLOAT, (4, 3))], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 1), make_opsetid(ONNX_DOMAIN, 11)])