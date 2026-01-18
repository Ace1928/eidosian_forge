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
def test_if_function(self):
    then_out = make_tensor_value_info('then_out', TensorProto.FLOAT, [5])
    else_out = make_tensor_value_info('else_out', TensorProto.FLOAT, [5])
    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    y = np.array([5, 4, 3, 2, 1]).astype(np.float32)
    then_const_node = make_node('Constant', inputs=[], outputs=['then_out'], value=from_array(x))
    else_const_node = make_node('Constant', inputs=[], outputs=['else_out'], value=from_array(y))
    then_body = make_graph([then_const_node], 'then_body', [], [then_out])
    else_body = make_graph([else_const_node], 'else_body', [], [else_out])
    if_node = make_node('If', inputs=['f_cond'], outputs=['f_res'], then_branch=then_body, else_branch=else_body)
    f = FunctionProto()
    f.domain = 'custom'
    f.name = 'fn'
    f.input.extend(['f_cond'])
    f.output.extend(['f_res'])
    f.node.extend([if_node])
    opset = onnx_opset_version()
    f.opset_import.extend([make_opsetid('', opset)])
    graph = make_graph(nodes=[make_node('fn', domain='custom', inputs=['cond'], outputs=['res'])], name='graph', inputs=[make_tensor_value_info('cond', TensorProto.BOOL, [])], outputs=[make_tensor_value_info('res', TensorProto.FLOAT, [5])])
    m = make_model(graph, producer_name='test', opset_imports=[make_opsetid('', opset), make_opsetid('custom', 1)])
    m.functions.extend([f])
    sess = ReferenceEvaluator(m)
    result = sess.run(None, {'cond': np.array(True)})
    expected = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    assert_allclose(expected, result[0])