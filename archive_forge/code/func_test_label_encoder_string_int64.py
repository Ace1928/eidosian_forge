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
@parameterized.expand(all_versions_for('LabelEncoder') if ONNX_ML else [], skip_on_empty=True)
def test_label_encoder_string_int64(self, _, version) -> None:
    self.skipIf(version < 2, 'keys_* attributes were introduced in ai.onnx.ml opset 2')
    string_list = ['A', 'm', 'y']
    float_list = [94.17, 36.0, -99.0]
    int64_list = [12, 28, 86]
    graph = self._make_graph([('x', TensorProto.STRING, (6, 1))], [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN, keys_strings=string_list, values_int64s=int64_list)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (6, 1))], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, version), make_opsetid(ONNX_DOMAIN, 11)])
    graph = self._make_graph([('x', TensorProto.INT64, (2, 3))], [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN, keys_int64s=int64_list, values_strings=string_list)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.STRING, (2, 3))], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, version), make_opsetid(ONNX_DOMAIN, 11)])
    graph = self._make_graph([('x', TensorProto.FLOAT, (2,))], [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN, keys_floats=float_list, values_int64s=int64_list)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.INT64, (2,))], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, version), make_opsetid(ONNX_DOMAIN, 11)])
    graph = self._make_graph([('x', TensorProto.INT64, (8,))], [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN, keys_int64s=int64_list, values_floats=float_list)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (8,))], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, version), make_opsetid(ONNX_DOMAIN, 11)])
    graph = self._make_graph([('x', TensorProto.FLOAT, ())], [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN, keys_floats=float_list, values_strings=string_list)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.STRING, ())], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, version), make_opsetid(ONNX_DOMAIN, 11)])
    graph = self._make_graph([('x', TensorProto.STRING, (1, 2))], [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN, keys_strings=string_list, values_floats=float_list)], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (1, 2))], opset_imports=[make_opsetid(ONNX_ML_DOMAIN, version), make_opsetid(ONNX_DOMAIN, 11)])