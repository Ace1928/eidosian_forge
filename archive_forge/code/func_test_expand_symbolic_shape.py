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
def test_expand_symbolic_shape(self, _, version) -> None:
    graph = self._make_graph([('x', TensorProto.INT32, (1, 2, None)), ('shape', TensorProto.INT64, ('unk__0',))], [make_node('Expand', ['x', 'shape'], ['y'])], [], initializer=[])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT32, None)], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])