import unittest
from typing import Sequence
from shape_inference_test import TestShapeInferenceHelper
import onnx
import onnx.helper
import onnx.parser
import onnx.shape_inference
from onnx import AttributeProto, TypeProto
def test_fi_optional_input(self):
    code = '\n            <opset_import: [ "" : 18 ], domain: "local">\n            DoReduce (x, axes) => (y) {\n                y = ReduceMax (x, axes)\n            }\n        '
    self._check(code, [float_type_], [], [float_type_])
    self._check(code, [float_type_, no_type_], [], [float_type_])
    code = '\n            <opset_import: [ "" : 18 ], domain: "local">\n            Quantize (x, scale, zero_point) => (y) {\n                y = QuantizeLinear (x, scale, zero_point)\n            }\n        '
    self._check(code, [float_type_, float_type_, int8_type_], [], [int8_type_])
    self._check(code, [float_type_, float_type_, uint8_type_], [], [uint8_type_])
    self._check(code, [float_type_, float_type_, no_type_], [], [uint8_type_])
    code = '\n            <opset_import: [ "" : 18 ], domain: "local">\n            DoClip (x, min, max) => (y) {\n                y = Clip (x, min, max)\n            }\n        '
    self._check(code, [float_type_, no_type_, float_type_], [], [float_type_])
    self._check_fails(code, [float_type_, no_type_, int8_type_], [])