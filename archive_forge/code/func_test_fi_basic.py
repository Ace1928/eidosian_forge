import unittest
from typing import Sequence
from shape_inference_test import TestShapeInferenceHelper
import onnx
import onnx.helper
import onnx.parser
import onnx.shape_inference
from onnx import AttributeProto, TypeProto
def test_fi_basic(self):
    code = '\n            <opset_import: [ "" : 18 ], domain: "local">\n            f (y, z) => (w) {\n                x = Add(y, z)\n                w = Mul(x, y)\n            }\n        '
    self._check(code, [float_type_, float_type_], [], [float_type_])
    self._check(code, [int32_type_, int32_type_], [], [int32_type_])
    self._check_fails(code, [float_type_, int32_type_], [])