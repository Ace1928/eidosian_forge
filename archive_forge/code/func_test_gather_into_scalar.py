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
@parameterized.expand(all_versions_for('Gather'))
def test_gather_into_scalar(self, _, version) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (3,)), ('i', TensorProto.INT64, ())], [make_node('Gather', ['x', 'i'], ['y'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, ())], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)])