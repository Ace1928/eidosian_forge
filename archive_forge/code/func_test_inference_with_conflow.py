import unittest
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import onnx
from onnx import TensorProto, TypeProto
from onnx.checker import ValidationError
from onnx.defs import OpSchema, get_all_schemas_with_history, get_schema
from onnx.helper import (
from onnx.numpy_helper import from_array
from onnx.shape_inference import InferenceError, infer_node_outputs
def test_inference_with_conflow(self) -> None:
    model_script = '\n        <\n            ir_version: 8,\n            opset_import: ["" : 18, "onnxscript.atenlib" : 1],\n            producer_name: "pytorch",\n            producer_version: "2.1.0"\n        >\n        torch_jit (float input_0) => (float reault, int64 index)\n        {\n            reault, index = onnxscript.atenlib.aten_min_dim <dim = 0, keepdim = 1> (input_0)\n        }\n        <\n            domain: "onnxscript.atenlib",\n            opset_import: ["" : 18]\n        >\n        aten_min_dim <dim>(self) => (result_7, indices_6)\n        {\n            tmp = Shape (self)\n            tmp_0 = Size (tmp)\n            tmp_1 = Constant <value = int64 tmp_1 {0}> ()\n            tmp_1_cast = CastLike (tmp_1, tmp_0)\n            tmp_2 = Equal (tmp_0, tmp_1_cast)\n            cond = Not (tmp_2)\n            indices_6, result_7 = If (cond) <\n                then_branch = thenGraph_4 () => ( indices,  result) {\n                    dim = Constant <value_int: int = @dim> ()\n                    tmp_3 = Constant <value_ints = [-1]> ()\n                    dims = Reshape (dim, tmp_3)\n                    result = ReduceMin <keepdims: int = @keepdim> (self, dims)\n                    indices = ArgMin <axis: int = @dim, keepdims: int = @keepdim> (self)\n                }, else_branch = elseGraph_4 () => ( indices_4,  result_5) {\n                    indices_4 = Constant <value_int = 0> ()\n                    result_5 = Identity (self)\n                }\n            >\n        }\n        '
    model = onnx.parser.parse_model(model_script)
    onnx.shape_inference.infer_shapes(model, strict_mode=False)
    with self.assertRaises(onnx.shape_inference.InferenceError):
        onnx.shape_inference.infer_shapes(model, strict_mode=True)