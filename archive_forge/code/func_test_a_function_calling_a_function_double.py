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
def test_a_function_calling_a_function_double(self):
    X = make_tensor_value_info('X', TensorProto.FLOAT, ['N'])
    output = make_tensor_value_info('output', TensorProto.FLOAT, ['N'])
    Z = make_tensor_value_info('output', TensorProto.FLOAT, ['N'])
    func_def_add = make_function('this', 'fctadd', ['input2'], ['output'], [make_node('Constant', [], ['one'], value_floats=[1.0], name='CC0'), make_node('Add', ['input2', 'one'], ['output'], name='A1')], opset_imports=[make_operatorsetid('', 15)])
    func_def = make_function('this', 'fct', ['input'], ['output'], [make_node('Constant', [], ['one'], value_floats=[1.0], name='CC'), make_node('Greater', ['input', 'one'], ['cond']), make_node('If', ['cond'], ['output'], then_branch=make_graph([make_node('fctadd', ['input'], ['output'], domain='this')], 'gthen', [], [output]), else_branch=make_graph([make_node('Add', ['input', 'one'], ['output'], domain='')], 'gelse', [], [output]), name=':IF')], opset_imports=[make_operatorsetid('', 15), make_operatorsetid('this', 1)])
    model_def = make_model(make_graph([make_node('fct', ['X'], ['ztmp'], domain='this'), make_node('fct', ['ztmp'], ['output'], domain='this')], 'test', [X], [Z]), ir_version=7, opset_imports=[make_operatorsetid('', 15), make_operatorsetid('this', 1)], functions=[func_def_add, func_def])
    feeds = {'X': np.array([-5], dtype=np.float32)}
    oinf = ReferenceEvaluator(model_def)
    expected = oinf.run(None, feeds)
    self.assertEqual(expected[0], np.array([-3], dtype=np.float32))