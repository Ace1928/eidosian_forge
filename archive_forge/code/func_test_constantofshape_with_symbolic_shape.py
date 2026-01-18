from __future__ import annotations
import unittest
from shape_inference_test import TestShapeInferenceHelper
import onnx.parser
from onnx import TensorProto
from onnx.helper import make_node, make_tensor, make_tensor_value_info
def test_constantofshape_with_symbolic_shape(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (3, 4, 5))], [make_node('Shape', ['x'], ['shape']), make_node('ConstantOfShape', ['shape'], ['y'], value=make_tensor('value', TensorProto.INT32, (1,), (2,)))], [])
    self._assert_inferred(graph, [make_tensor_value_info('shape', TensorProto.INT64, (3,)), make_tensor_value_info('y', TensorProto.INT32, (3, 4, 5))], data_prop=True)