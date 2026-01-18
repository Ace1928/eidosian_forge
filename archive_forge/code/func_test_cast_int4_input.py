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
@parameterized.parameterized.expand(itertools.product((TensorProto.UINT4, TensorProto.INT4), (TensorProto.FLOAT, TensorProto.FLOAT16)))
def test_cast_int4_input(self, cast_from, cast_to):
    X = make_tensor_value_info('X', cast_from, [None])
    Y = make_tensor_value_info('Y', cast_to, [None])
    model = make_model(make_graph([make_node('Cast', ['X'], ['Y'], to=TensorProto.FLOAT)], 'g', [X], [Y]))
    ref = ReferenceEvaluator(model)
    data = np.array(range(0, 7), dtype=np.float32)
    cast_from_np = custom.uint4 if cast_from == TensorProto.UINT4 else custom.int4
    data = data.astype(cast_from_np)
    expected1 = np.array([subbyte.float32_to_4bit_unpacked(x, cast_from_np) for x in data])
    got = ref.run(None, {'X': data})
    self.assertEqual(expected1.tolist(), got[0].tolist())