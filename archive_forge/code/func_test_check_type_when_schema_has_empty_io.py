from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
def test_check_type_when_schema_has_empty_io(self):
    input = '\n            <\n                ir_version: 7,\n                opset_import: ["" : 1]\n            >\n            agraph (X, Y) => (Z)\n            {\n                Z = CustomOp(X, Y)\n            }\n           '
    model = onnx.parser.parse_model(input)
    op_schema = defs.OpSchema('CustomOp', '', 1, inputs=[], outputs=[])
    onnx.defs.register_schema(op_schema)
    with self.assertRaises(onnx.shape_inference.InferenceError):
        onnx.shape_inference.infer_shapes(model, True)
    onnx.defs.deregister_schema(op_schema.name, op_schema.since_version, op_schema.domain)