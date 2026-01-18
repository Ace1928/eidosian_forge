import itertools
import math
import sys
import unittest
from contextlib import redirect_stdout
from functools import wraps
from io import StringIO
from os import getenv
from textwrap import dedent
from typing import Sequence, Tuple
import numpy as np
import parameterized
import version_utils
from numpy.testing import assert_allclose
import onnx.reference.custom_element_types as custom
from onnx import (
from onnx.backend.test.case.node.roialign import get_roi_align_input_values
from onnx.checker import check_model
from onnx.defs import onnx_opset_version
from onnx.helper import (
from onnx.numpy_helper import float8e4m3_to_float32, float8e5m2_to_float32, from_array
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun, OpRunExpand
from onnx.reference.ops import load_op
from onnx.reference.ops._op_common_indices import _get_indices, _is_out
from onnx.reference.ops._op_list import Cast_19, Celu
from onnx.reference.ops.aionnx_preview_training._op_list import Adam
from onnx.reference.ops.op_celu import _vcelu1
from onnx.reference.ops.op_col2im import (
from onnx.reference.ops.op_conv import Conv, _conv_implementation
from onnx.reference.ops_optimized import Conv as ConvOptimized
from onnx.reference.ops_optimized.op_conv_optimized import _conv_implementation_im2col
def test_nested_local_functions(self):
    m = parser.parse_model('\n            <\n              ir_version: 8,\n              opset_import: [ "" : 14, "local" : 1],\n              producer_name: "test",\n              producer_version: "1.0",\n              model_version: 1,\n              doc_string: "Test preprocessing model"\n            >\n            agraph (uint8[H, W, C] x) => (uint8[H, W, C] x_processed)\n            {\n                x_processed = local.func(x)\n            }\n\n            <\n              opset_import: [ "" : 14 ],\n              domain: "local",\n              doc_string: "function 1"\n            >\n            f1 (x) => (y) {\n                y = Identity(x)\n            }\n\n            <\n              opset_import: [ "" : 14 ],\n              domain: "local",\n              doc_string: "function 2"\n            >\n            f2 (x) => (y) {\n                y = Identity(x)\n            }\n\n            <\n              opset_import: [ "" : 14, "local" : 1 ],\n              domain: "local",\n              doc_string: "Preprocessing function."\n            >\n            func (x) => (y) {\n                x1 = local.f1(x)\n                y = local.f2(x1)\n            }\n        ')
    sess = ReferenceEvaluator(m)
    x = np.array([0, 1, 3], dtype=np.uint8).reshape((1, 1, 3))
    result = sess.run(None, {'x': x})[0]
    expected = x
    assert_allclose(expected, result)