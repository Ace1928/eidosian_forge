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
@parameterized.expand(all_versions_for('Expand'))
def test_expand_raw_data(self, _, version) -> None:
    graph = self._make_graph([('x', TensorProto.INT32, (3, 1)), ('shape', TensorProto.INT64, (2,))], [make_node('Expand', ['x', 'shape'], ['y'])], [], initializer=[make_tensor('shape', TensorProto.INT64, (2,), vals=np.array([3, 4], dtype='<i8').tobytes(), raw=True)])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT32, (3, 4))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])