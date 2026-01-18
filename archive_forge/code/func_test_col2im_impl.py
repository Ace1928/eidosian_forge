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
def test_col2im_impl(self):

    def get_im2col_indices(x_shape, field_height, field_width, padding=None, stride=1):
        N, C, H, W = x_shape
        del N
        assert (H + padding[0] + padding[2] - field_height) % stride == 0
        assert (W + padding[1] + padding[3] - field_height) % stride == 0
        out_height = (H + padding[0] + padding[2] - field_height) // stride + 1
        out_width = (W + padding[1] + padding[3] - field_width) // stride + 1
        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
        return (k, i, j)

    def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=None, stride=1):
        N, C, H, W = x_shape
        H_padded, W_padded = (H + padding[0] + padding[2], W + padding[1] + padding[3])
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        padding = padding.copy()
        if padding[2] == 0:
            padding[2] += x_padded.shape[2]
        elif padding[2] > 0:
            padding[2] *= -1
        if padding[3] == 0:
            padding[3] += x_padded.shape[3]
        elif padding[3] > 0:
            padding[3] *= -1
        res = x_padded[:, :, padding[0]:padding[2], padding[1]:padding[3]]
        return res
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None, None])
    IS = make_tensor_value_info('IS', TensorProto.INT64, [None])
    BS = make_tensor_value_info('BS', TensorProto.INT64, [None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None, None, None, None])
    node = make_node('Col2Im', ['X', 'IS', 'BS'], ['Y'], pads=[0, 1, 0, 1])
    graph = make_graph([node], 'g', [X, IS, BS], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('', 16)])
    feeds = {'X': np.arange(5 * 15).astype(np.float32).reshape((1, 5, 15)), 'IS': np.array([5, 5]), 'BS': np.array([1, 5])}
    ref1 = ReferenceEvaluator(onnx_model)
    got1 = ref1.run(None, feeds)
    expected = col2im_indices(feeds['X'], (1, 1, 5, 5), field_height=1, field_width=5, padding=[0, 1, 0, 1])
    assert_allclose(expected, got1[0])