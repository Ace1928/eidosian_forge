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
def test_concat_in_a_function(self):

    def create_model():
        nodes = []
        inputs = []
        outputs = []
        functions = []
        opsets = {'': onnx_opset_version(), 'custom_domain': 1}
        nodes_fct = []
        node = make_node('Concat', ['x:0', 'x:1'], ['r__0'], axis=0, domain='')
        nodes_fct.append(node)
        opset_imports_fct = [make_opsetid(domain, 1 if version is None else version) for domain, version in opsets.items()]
        fct = make_function('custom_domain', 'concat_2', ['x:0', 'x:1'], ['r__0'], nodes_fct, opset_imports_fct)
        functions.append(fct)
        inputs.append(make_tensor_value_info('I__0', TensorProto.DOUBLE, []))
        inputs.append(make_tensor_value_info('I__1', TensorProto.DOUBLE, []))
        inputs.append(make_tensor_value_info('I__2', TensorProto.DOUBLE, []))
        outputs.append(make_tensor_value_info('r__4', TensorProto.DOUBLE, []))
        node = make_node('concat_2', ['I__0', 'I__1'], ['r__3'], axis=0, domain='custom_domain')
        nodes.append(node)
        node = make_node('concat_2', ['I__2', 'r__3'], ['r__4'], axis=0, domain='custom_domain')
        nodes.append(node)
        opset_imports = [make_opsetid(domain, 1 if version is None else version) for domain, version in opsets.items()]
        graph = make_graph(nodes, 'numpyx', inputs, outputs)
        onnx_model = make_model(graph, opset_imports=opset_imports, functions=functions)
        return onnx_model
    onnx_model = create_model()
    x1 = np.array([[-5, 6], [15, 3]], dtype=np.float64)
    x2 = np.array([[1, 2]], dtype=np.float64)
    x3 = np.array([[-1, -2]], dtype=np.float64)
    z = np.vstack([x1, x2, x3])
    ref = ReferenceEvaluator(onnx_model)
    feeds = {'I__2': x1, 'I__0': x2, 'I__1': x3}
    got = ref.run(None, feeds)
    assert_allclose(z, got[0])