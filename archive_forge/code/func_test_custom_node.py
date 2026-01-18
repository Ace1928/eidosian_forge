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
def test_custom_node(self):

    class _InvAlpha:
        op_domain = 'custom'

        def __init__(self, onnx_node, run_params):
            self.onnx_node = onnx_node
            self.run_params = run_params

        def _run(self, x):
            return (1 / (x + self.alpha),)

    class InvAlpha2(OpRun):

        def _run(self, x):
            return (1 / (x + self.alpha),)

    class InvAlpha(OpRun):
        op_domain = 'custom'

        def _run(self, x, alpha=None):
            alpha = alpha or self.alpha
            return (1 / (x + alpha),)
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])
    node1 = make_node('InvAlpha', ['X'], ['Y'], alpha=0.5, domain='custom')
    graph = make_graph([node1], 'rs', [X], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('custom', 1)])
    x = np.arange(60).reshape((3, 4, 5)).astype(np.float32) + 1
    with self.assertRaises(NotImplementedError):
        ReferenceEvaluator(onnx_model)
    node1 = make_node('_InvAlpha', ['X'], ['Y'], alpha=0.5, domain='custom')
    graph = make_graph([node1], 'rs', [X], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('custom', 1)])
    with self.assertRaises(TypeError):
        ReferenceEvaluator(onnx_model, new_ops=[_InvAlpha])
    node1 = make_node('InvAlpha2', ['X'], ['Y'], alpha=0.5, domain='custom')
    graph = make_graph([node1], 'rs', [X], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('custom', 1)])
    with self.assertRaises(NotImplementedError):
        ReferenceEvaluator(onnx_model, new_ops=[InvAlpha2])
    node1 = make_node('InvAlpha', ['X'], ['Y'], alpha=0.5, domain='custom')
    graph = make_graph([node1], 'rs', [X], [Y])
    onnx_model = make_model(graph, opset_imports=[make_opsetid('custom', 1)])
    sess = ReferenceEvaluator(onnx_model, new_ops=[InvAlpha, InvAlpha])
    got = sess.run(None, {'X': x})[0]
    expected = 1 / (x + 0.5)
    assert_allclose(expected, got)