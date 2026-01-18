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
def test_blackmanwindow(self):
    graph = self._make_graph([], [make_node('Constant', [], ['shape'], value=make_tensor('shape', TensorProto.INT64, (), (10,))), make_node('BlackmanWindow', ['shape'], ['y'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('shape', TensorProto.INT64, ()), make_tensor_value_info('y', TensorProto.FLOAT, (10,))])
    graph = self._make_graph([], [make_node('Constant', [], ['shape'], value=make_tensor('shape', TensorProto.INT64, (), (10,))), make_node('BlackmanWindow', ['shape'], ['y'], periodic=0)], [])
    self._assert_inferred(graph, [make_tensor_value_info('shape', TensorProto.INT64, ()), make_tensor_value_info('y', TensorProto.FLOAT, (10,))])