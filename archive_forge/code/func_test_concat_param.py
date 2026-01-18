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
def test_concat_param(self, _, version) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, ('a', 2)), ('y', TensorProto.FLOAT, ('a', 3))], [make_node('Concat', ['x', 'y'], ['z'], axis=1)], [])
    self._assert_inferred(graph, [make_tensor_value_info('z', TensorProto.FLOAT, ('a', 5))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])