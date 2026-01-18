import typing
import unittest
import onnx
import onnx.parser
import onnx.shape_inference
def test_mi_constant_in_function(self):
    model = '\n            <\n                ir_version: 7,\n                opset_import: [ "" : 17, "local" : 1]\n            >\n            main (float x) => (y, z) {\n                y, z = local.expand(x)\n            }\n            <\n                opset_import: [ "" : 17 ],\n                domain: "local"\n            >\n            expand (x) => (y, z) {\n                shape1 = Constant<value = int64[2] {4,4}>()\n                shape2 = Constant<value = int64[3] {8,8,8}>()\n                z = Expand (x, shape2)\n                y = Expand (x, shape1)\n            }\n            '
    self._check_shape(model, [4, 4], [8, 8, 8])