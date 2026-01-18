import typing
import unittest
import onnx
import onnx.parser
import onnx.shape_inference
def test_mi_function(self):
    """Test use of functions."""
    model = '\n            <\n                ir_version: 7,\n                opset_import: [ "" : 17, "local" : 1]\n            >\n            agraph (float[N] x) => (y)\n            {\n                y = local.cast(x)\n            }\n            <\n                opset_import: [ "" : 17 ],\n                domain: "local"\n            >\n            cast (x) => (y)\n            {\n                y = Cast<to=6> (x)\n            }\n        '
    self._check(model, onnx.TensorProto.INT32)