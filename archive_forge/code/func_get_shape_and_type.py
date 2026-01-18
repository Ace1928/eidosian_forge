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
def get_shape_and_type(matrix):
    """Return matrix's shape and type if available."""
    handle_data = getattr(matrix, '_handle_data', None)
    if handle_data is None:
        return None
    if len(handle_data.shape_and_type) != 1:
        raise ValueError('shape_and_type array in _handle_data must have length one, but saw: %d' % len(handle_data.shape_and_type))
    return handle_data.shape_and_type[0]