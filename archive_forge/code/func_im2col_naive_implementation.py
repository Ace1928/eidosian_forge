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
def im2col_naive_implementation(data, kernel_shape, dilations, pads, strides):
    """Naive implementation for `im2col`.

    Args:
        data: image (float)
        kernel_shape: kernel shape
        dilations: dilations
        pads: pads
        strides: strides

    Returns:
        result
    """
    if not isinstance(kernel_shape, tuple):
        raise TypeError(f'Unexpected type {type(kernel_shape)!r} for kernel_shape.')
    if len(data.shape) != len(kernel_shape):
        raise ValueError(f'Shape mismatch {data.shape!r} and {kernel_shape!r}.')
    n_dims = len(pads) // 2
    new_pads = np.array([(pads[i], pads[i + n_dims]) for i in range(n_dims)])
    list_output_shape = list(data.shape + kernel_shape)
    for d in range(n_dims):
        kd = kernel_shape[d] + (kernel_shape[d] - 1) * (dilations[d] - 1)
        nd = int((list_output_shape[d] - kd + new_pads[d][0] + new_pads[d][1]) / strides[d] + 1)
        list_output_shape[d] = nd
    output_shape = tuple(list_output_shape)
    res = np.zeros(output_shape, dtype=data.dtype)
    kernel_size = np.prod(kernel_shape)
    res_size = np.prod(res.shape[:-n_dims])
    for i in range(res_size):
        i_res = _get_indices(i, res.shape[:-n_dims])
        t_res = tuple(i_res)
        for j in range(kernel_size):
            i_kernel = _get_indices(j, kernel_shape)
            t_kernel = tuple(i_kernel)
            i_img = i_res * strides - new_pads[:, 0] + i_kernel * dilations
            t_img = tuple(i_img)
            if _is_out(t_img, data.shape):
                res[t_res + t_kernel] = 0
            else:
                res[t_res + t_kernel] = data[tuple(t_img)]
    return res