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
def test_dict_vectorizer(self):
    value_type = TypeProto()
    value_type.tensor_type.elem_type = TensorProto.INT64
    onnx_type = TypeProto()
    onnx_type.map_type.key_type = TensorProto.STRING
    onnx_type.map_type.value_type.CopyFrom(value_type)
    value_info = ValueInfoProto()
    value_info.name = 'X'
    value_info.type.CopyFrom(onnx_type)
    X = value_info
    Y = make_tensor_value_info('Y', TensorProto.INT64, [None, None])
    node1 = make_node('DictVectorizer', ['X'], ['Y'], domain='ai.onnx.ml', string_vocabulary=['a', 'c', 'b', 'z'])
    graph = make_graph([node1], 'ml', [X], [Y])
    onx = make_model_gen_version(graph, opset_imports=OPSETS)
    check_model(onx)
    x = {'a': np.array(4, dtype=np.int64), 'c': np.array(8, dtype=np.int64)}
    expected = np.array([4, 8, 0, 0], dtype=np.int64)
    sess = ReferenceEvaluator(onx)
    got = sess.run(None, {'X': x})[0]
    self.assertEqual(expected.tolist(), got.tolist())