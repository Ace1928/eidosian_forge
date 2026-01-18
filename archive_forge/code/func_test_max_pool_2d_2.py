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
def test_max_pool_2d_2(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
    node = make_node('MaxPool', ['X'], ['Y'], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2])
    graph = make_graph([node], 'g', [X], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
    feeds = {'X': np.array([[[[683, 358, 726, 578, 650, 946, 200], [679, 260, 264, 5, 240, 255, 582], [322, 66, 687, 632, 852, 698, 428], [111, 452, 627, 332, 751, 842, 685], [472, 52, 956, 81, 807, 827, 360], [972, 574, 81, 799, 646, 499, 486], [892, 758, 75, 833, 972, 415, 736]]]], dtype=np.float32)}
    expected = np.array([[[[683.0, 726.0, 946.0, 946.0], [679.0, 687.0, 852.0, 842.0], [972.0, 956.0, 842.0, 842.0], [972.0, 833.0, 972.0, 736.0]]]], dtype=np.float32)
    ref1 = ReferenceEvaluator(onnx_model)
    got1 = ref1.run(None, feeds)
    assert_allclose(expected, got1[0])