import typing
import unittest
import onnx
import onnx.parser
import onnx.shape_inference
def test_mi_constant_2(self):
    model = '\n            <\n                ir_version: 7,\n                opset_import: [ "" : 17]\n            >\n            mymodel (float[4, 8, 16] x) => (y) {\n                shape = Constant<value_ints=[4,2,8]>()\n                two = Constant<value_int=2>()\n                shape2 = Mul(shape, two)\n                y = Reshape(x, shape2)\n            }\n            '
    self._check_shape(model, [8, 4, 16])