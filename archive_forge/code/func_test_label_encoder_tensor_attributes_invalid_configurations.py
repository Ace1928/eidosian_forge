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
def test_label_encoder_tensor_attributes_invalid_configurations(self, _, version) -> None:
    self.skipIf(version < 4, 'tensor attributes introduced in ai.onnx.ml opset 4')
    key_tensor = make_tensor('keys_tensor', TensorProto.STRING, [4], ['a', 'b', 'cc', 'ddd'])
    values_tensor = make_tensor('values_tensor', TensorProto.INT64, [4], [1, 2, 3, 4])
    opset_imports = [make_opsetid(ONNX_ML_DOMAIN, version), make_opsetid(ONNX_DOMAIN, 11)]
    graph = self._make_graph([('x', TensorProto.STRING, ('M', None, 3, 12))], [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN, keys_tensor=key_tensor, values_tensor=values_tensor, default_tensor=make_tensor('default_tensor', TensorProto.STRING, [1], [0]))], [])
    self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph, opset_imports=opset_imports)
    graph = self._make_graph([('x', TensorProto.STRING, ('M', None, 3, 12))], [make_node('LabelEncoder', ['x'], ['y'], domain=ONNX_ML_DOMAIN, keys_tensor=key_tensor, values_strings=['a', 'b', 'cc', 'ddd'], default_tensor=make_tensor('default_tensor', TensorProto.STRING, [1, 2], [0, 0]))], [])
    self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph, opset_imports=opset_imports)