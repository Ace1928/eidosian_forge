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
def test_expand_scalar_input(self, _, version) -> None:
    graph = self._make_graph([('x', TensorProto.INT32, ()), ('shape', TensorProto.INT64, (2,))], [make_node('Expand', ['x', 'shape'], ['y'])], [], initializer=[make_tensor('shape', TensorProto.INT64, (2,), (4, 8))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT32, (4, 8))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])