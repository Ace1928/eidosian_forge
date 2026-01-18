import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_empty_list_attribute(self):
    model = onnx.parser.parse_model('\n            <\n                ir_version: 7,\n                opset_import: [ "" : 17]\n            >\n            agraph (float[N] x) => (int64[M] y)\n            {\n                y = Constant <value_ints: ints = []>()\n            }\n        ')
    checker.check_model(model, full_check=True)
    model = onnx.parser.parse_model('\n            <\n                ir_version: 7,\n                opset_import: [ "" : 17]\n            >\n            agraph (float[N] x) => (float[M] y)\n            {\n                y = Constant <value_floats: floats = []>()\n            }\n        ')
    checker.check_model(model, full_check=True)