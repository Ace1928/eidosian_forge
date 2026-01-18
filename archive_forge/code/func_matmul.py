from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops
def matmul(x, y, **kwargs):
    return _PrunedDenseMatrixMultiplication(x, y, indices=sparse_csr_matrix_ops.csr_sparse_matrix_to_sparse_tensor(a, type=x.dtype).indices, **kwargs)