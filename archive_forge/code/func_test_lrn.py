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
def test_lrn(self):

    def _expected(x, alpha, beta, bias, size):
        square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
        for n, c, h, w in np.ndindex(x.shape):
            square_sum[n, c, h, w] = sum(x[n, max(0, c - int(math.floor((size - 1) / 2))):min(5, c + int(math.ceil((size - 1) / 2)) + 1), h, w] ** 2)
        y = x / (bias + alpha / size * square_sum) ** beta
        return y
    alpha = 0.0002
    beta = 0.5
    bias = 2.0
    size = 3
    X = make_tensor_value_info('X', TensorProto.FLOAT, [5, 5, 50, 50])
    Z = make_tensor_value_info('Z', TensorProto.UNDEFINED, None)
    nodes = [make_node('LRN', ['X'], ['Z'], alpha=alpha, beta=beta, bias=bias, size=size)]
    model = make_model(make_graph(nodes, 'g', [X], [Z]))
    ref = ReferenceEvaluator(model)
    data = np.random.rand(5, 5, 5, 5).astype(np.float32)
    got = ref.run(None, {'X': data})
    expected = _expected(data, alpha, beta, bias, size)
    self.assertEqual(len(expected), len(got[0]))