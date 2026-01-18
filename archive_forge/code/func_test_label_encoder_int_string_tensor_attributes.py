import itertools
import unittest
from functools import wraps
from os import getenv
import numpy as np  # type: ignore
from numpy.testing import assert_allclose  # type: ignore
from parameterized import parameterized
import onnx
from onnx import ONNX_ML, TensorProto, TypeProto, ValueInfoProto
from onnx.checker import check_model
from onnx.defs import onnx_ml_opset_version, onnx_opset_version
from onnx.helper import (
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops.aionnxml.op_tree_ensemble import (
@unittest.skipIf(not ONNX_ML, reason='onnx not compiled with ai.onnx.ml')
def test_label_encoder_int_string_tensor_attributes(self):
    X = make_tensor_value_info('X', TensorProto.INT64, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.STRING, [None, None])
    node = make_node('LabelEncoder', ['X'], ['Y'], domain='ai.onnx.ml', keys_tensor=make_tensor('keys_tensor', TensorProto.INT64, [4], [1, 2, 3, 4]), values_tensor=make_tensor('values_tensor', TensorProto.STRING, [4], ['a', 'b', 'cc', 'ddd']), default_tensor=make_tensor('default_tensor', TensorProto.STRING, [], ['NONE']))
    graph = make_graph([node], 'ml', [X], [Y])
    model = make_model_gen_version(graph, opset_imports=OPSETS)
    check_model(model)
    x = np.array([[0, 1, 3, 4]], dtype=np.int64).T
    expected = np.array([['NONE'], ['a'], ['cc'], ['ddd']])
    self._check_ort(model, {'X': x}, equal=True)
    sess = ReferenceEvaluator(model)
    got = sess.run(None, {'X': x})[0]
    self.assertEqual(expected.tolist(), got.tolist())