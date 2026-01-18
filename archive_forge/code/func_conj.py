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
def conj(self):
    return self._from_matrix(math_ops.conj(self._matrix), self.eager_handle_data)