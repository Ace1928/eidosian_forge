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
def test_category_mapper(self) -> None:
    cat = make_node('CategoryMapper', ['x'], ['y'], domain=ONNX_ML_DOMAIN)
    graph_int = self._make_graph([('x', TensorProto.INT64, (30, 4, 5))], [cat], [])
    self._assert_inferred(graph_int, [make_tensor_value_info('y', TensorProto.STRING, (30, 4, 5))], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 1), make_opsetid(ONNX_DOMAIN, 11)])
    graph_str = self._make_graph([('x', TensorProto.STRING, (30, 5, 4))], [cat], [])
    self._assert_inferred(graph_str, [make_tensor_value_info('y', TensorProto.INT64, (30, 5, 4))], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 1), make_opsetid(ONNX_DOMAIN, 11)])