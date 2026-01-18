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
def test_imputer_float_2d(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    node1 = make_node('Imputer', ['X'], ['Y'], domain='ai.onnx.ml', imputed_value_floats=np.array([0, 0.1], dtype=np.float32), replaced_value_float=np.nan)
    graph = make_graph([node1], 'ml', [X], [Y])
    onx = make_model_gen_version(graph, opset_imports=OPSETS)
    check_model(onx)
    x = np.array([[0, 1, np.nan, 3], [0, 1, np.nan, 3]], dtype=np.float32).T
    expected = np.array([[0, 1, 0, 3], [0, 1, 0.1, 3]], dtype=np.float32).T
    self._check_ort(onx, {'X': x})
    sess = ReferenceEvaluator(onx)
    got = sess.run(None, {'X': x})[0]
    assert_allclose(expected, got)