import abc
import collections
from tensorflow.python.eager import context
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.linalg.sparse import gen_sparse_csr_matrix_ops as sm_ops
from tensorflow.python.ops.linalg.sparse.gen_sparse_csr_matrix_ops import *
def matmul_shape_inference(a, b, c, transpose_a, transpose_b, adjoint_a, adjoint_b):
    """Helper function for matmul to set the result matrix's handle data."""
    c_handle = getattr(c, '_handle_data', None)
    a_shape_and_type = get_shape_and_type(a)
    b_shape_and_type = get_shape_and_type(b)
    if c_handle is None and a_shape_and_type is not None and (b_shape_and_type is not None):
        transpose_a = transpose_a or adjoint_a
        transpose_b = transpose_b or adjoint_b
        a_shape = a_shape_and_type.shape
        b_shape = b_shape_and_type.shape
        rank = len(a_shape.dim)
        c_rows = a_shape.dim[rank - (1 if transpose_a else 2)].size
        c_cols = b_shape.dim[rank - (2 if transpose_b else 1)].size
        c_shape = tensor_shape.TensorShape(a_shape)
        c_shape = tensor_shape.TensorShape(c_shape[:rank - 2] + [c_rows, c_cols])
        c_handle = _create_handle_data_proto(c_shape.as_proto(), a_shape_and_type.dtype)
    return c_handle