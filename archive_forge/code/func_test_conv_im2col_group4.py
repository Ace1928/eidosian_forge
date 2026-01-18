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
def test_conv_im2col_group4(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, [2, 4, 6, 6])
    W = make_tensor_value_info('W', TensorProto.FLOAT, [4, 1, 3, 3])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [4])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [2, 4, 6, 6])
    node = make_node('Conv', ['X', 'W', 'B'], ['Y'], group=4, dilations=[1, 1], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    graph = make_graph([node], 'g', [X, W, B], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
    feeds = {'X': np.arange(2 * 4 * 6 * 6).reshape((2, 4, 6, 6)).astype(np.float32), 'W': np.array([[[[-0.026239916682243347, 0.07565222680568695, -0.03209298849105835], [-0.08708783239126205, 0.0961190015077591, 0.13418219983577728], [0.1598859578371048, 0.03840477764606476, -0.13170936703681946]]], [[[-0.0689004510641098, 0.1408083587884903, -0.03717087209224701], [0.030967697501182556, 0.0263785719871521, -0.0899493545293808], [0.07828782498836517, -0.06266771256923676, 0.10750330984592438]]], [[[0.020227551460266113, -0.04353883117437363, -0.10938453674316406], [-0.14101561903953552, -0.03393106162548065, 0.12139306962490082], [0.02838282287120819, 0.13864465057849884, -0.06065710633993149]]], [[[-0.06511610746383667, -0.05987360328435898, -0.008047685027122498], [0.07340313494205475, 0.0326494425535202, 0.012516498565673828], [0.13260947167873383, -0.022225692868232727, -0.11167611926794052]]]], dtype=np.float32), 'B': np.array([-0.1457933485507965, -0.07481209933757782, -0.05890338122844696, -0.11964251846075058], dtype=np.float32)}
    feeds['B'][:] = 0
    X = feeds['X']
    W = feeds['W']
    B = feeds['B']
    Y = np.empty((2, 4, 6, 6), dtype=X.dtype)
    for b in range(X.shape[0]):
        for g in range(4):
            x = X[b:b + 1, g:g + 1]
            w = W[g]
            c2 = im2col(x, (3, 3), [1, 1], [1, 1, 1, 1], [1, 1])
            mul = np.matmul(c2, w.flatten())
            mul = mul + B[g]
            Y[b, g, :, :] = mul
    ref1 = ReferenceEvaluator(onnx_model)
    got1 = ref1.run(None, feeds)
    assert_allclose(Y, got1[0], atol=1e-05)