import typing
import unittest
import onnx
import onnx.parser
import onnx.shape_inference
def test_mi_function_default_attr(self):
    """Test use of default values of function attributes."""
    model = '\n            <ir_version: 7, opset_import: [ "" : 17, "local" : 1]>\n            agraph (float[N] x) => (y, z)\n            {\n                y = local.cast <target=6> (x) # casts to INT32 type (encoding value 6)\n                z = local.cast (x)  # uses default-attribute value of 1 (FLOAT type)\n            }\n\n            <opset_import: [ "" : 17 ], domain: "local">\n            cast <target: int = 1> (x) => (y)\n            {\n                y = Cast <to:int = @target> (x)\n            }\n        '
    self._check(model, onnx.TensorProto.INT32, onnx.TensorProto.FLOAT)