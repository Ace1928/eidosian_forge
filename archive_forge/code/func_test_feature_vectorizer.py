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
def test_feature_vectorizer(self):
    X = [make_tensor_value_info('X0', TensorProto.FLOAT, [None, None]), make_tensor_value_info('X1', TensorProto.FLOAT, [None, None])]
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None])
    x = [np.arange(9).reshape((3, 3)).astype(np.float32), np.arange(9).reshape((3, 3)).astype(np.float32) + 0.5]
    expected = {(1,): np.array([[0], [3], [6]], dtype=np.float32), (2,): np.array([[0, 1], [3, 4], [6, 7]], dtype=np.float32), (4,): np.array([[0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0]], dtype=np.float32), (1, 1): np.array([[0, 0.5], [3, 3.5], [6, 6.5]], dtype=np.float32), (0, 1): np.array([[0.5], [3.5], [6.5]], dtype=np.float32)}
    for inputdimensions, value in expected.items():
        att = list(inputdimensions) if isinstance(inputdimensions, tuple) else inputdimensions
        with self.subTest(inputdimensions=att):
            node1 = make_node('FeatureVectorizer', [f'X{i}' for i in range(len(att))], ['Y'], inputdimensions=att, domain='ai.onnx.ml')
            graph = make_graph([node1], 'ml', X[:len(att)], [Y])
            onx = make_model_gen_version(graph, opset_imports=OPSETS)
            check_model(onx)
            feeds = {f'X{i}': v for i, v in enumerate(x[:len(att)])}
            self._check_ort(onx, feeds, atol=1e-06)
            sess = ReferenceEvaluator(onx)
            got = sess.run(None, feeds)[0]
            assert_allclose(value, got, atol=1e-06)