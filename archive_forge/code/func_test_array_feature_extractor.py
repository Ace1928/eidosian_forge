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
def test_array_feature_extractor(self) -> None:
    node = make_node('ArrayFeatureExtractor', ['x', 'y'], ['z'], domain=ONNX_ML_DOMAIN)
    for axes_shape, expected in [((2,), 2), ((), 'unk__0'), (('N',), 'N')]:
        graph = self._make_graph([('x', TensorProto.INT64, (3, 4, 5)), ('y', TensorProto.INT64, axes_shape)], [node], [])
        self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.INT64, (3, 4, expected))], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, 3), make_opsetid(ONNX_DOMAIN, 18)])