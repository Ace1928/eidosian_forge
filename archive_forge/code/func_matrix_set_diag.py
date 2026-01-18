import numbers
import numpy as np
from tensorflow.core.config import flags
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import shape_util
from tensorflow.python.ops.gen_array_ops import *
from tensorflow.python.ops.gen_array_ops import reverse_v2 as reverse  # pylint: disable=unused-import
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
@tf_export('linalg.set_diag', v1=['linalg.set_diag', 'matrix_set_diag'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('matrix_set_diag')
def matrix_set_diag(input, diagonal, name='set_diag', k=0, align='RIGHT_LEFT'):
    """Returns a batched matrix tensor with new batched diagonal values.

  Given `input` and `diagonal`, this operation returns a tensor with the
  same shape and values as `input`, except for the specified diagonals of the
  innermost matrices. These will be overwritten by the values in `diagonal`.

  `input` has `r+1` dimensions `[I, J, ..., L, M, N]`. When `k` is scalar or
  `k[0] == k[1]`, `diagonal` has `r` dimensions `[I, J, ..., L, max_diag_len]`.
  Otherwise, it has `r+1` dimensions `[I, J, ..., L, num_diags, max_diag_len]`.
  `num_diags` is the number of diagonals, `num_diags = k[1] - k[0] + 1`.
  `max_diag_len` is the longest diagonal in the range `[k[0], k[1]]`,
  `max_diag_len = min(M + min(k[1], 0), N + min(-k[0], 0))`

  The output is a tensor of rank `k+1` with dimensions `[I, J, ..., L, M, N]`.
  If `k` is scalar or `k[0] == k[1]`:

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, n-max(k[1], 0)] ; if n - m == k[1]
      input[i, j, ..., l, m, n]              ; otherwise
  ```

  Otherwise,

  ```
  output[i, j, ..., l, m, n]
    = diagonal[i, j, ..., l, diag_index, index_in_diag] ; if k[0] <= d <= k[1]
      input[i, j, ..., l, m, n]                         ; otherwise
  ```
  where `d = n - m`, `diag_index = k[1] - d`, and
  `index_in_diag = n - max(d, 0) + offset`.

  `offset` is zero except when the alignment of the diagonal is to the right.
  ```
  offset = max_diag_len - diag_len(d) ; if (`align` in {RIGHT_LEFT, RIGHT_RIGHT}
                                             and `d >= 0`) or
                                           (`align` in {LEFT_RIGHT, RIGHT_RIGHT}
                                             and `d <= 0`)
           0                          ; otherwise
  ```
  where `diag_len(d) = min(cols - max(d, 0), rows + min(d, 0))`.

  For example:

  ```
  # The main diagonal.
  input = np.array([[[7, 7, 7, 7],              # Input shape: (2, 3, 4)
                     [7, 7, 7, 7],
                     [7, 7, 7, 7]],
                    [[7, 7, 7, 7],
                     [7, 7, 7, 7],
                     [7, 7, 7, 7]]])
  diagonal = np.array([[1, 2, 3],               # Diagonal shape: (2, 3)
                       [4, 5, 6]])
  tf.matrix_set_diag(input, diagonal)
    ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
          [7, 2, 7, 7],
          [7, 7, 3, 7]],
         [[4, 7, 7, 7],
          [7, 5, 7, 7],
          [7, 7, 6, 7]]]

  # A superdiagonal (per batch).
  tf.matrix_set_diag(input, diagonal, k = 1)
    ==> [[[7, 1, 7, 7],  # Output shape: (2, 3, 4)
          [7, 7, 2, 7],
          [7, 7, 7, 3]],
         [[7, 4, 7, 7],
          [7, 7, 5, 7],
          [7, 7, 7, 6]]]

  # A band of diagonals.
  diagonals = np.array([[[9, 1, 0],  # Diagonal shape: (2, 4, 3)
                         [6, 5, 8],
                         [1, 2, 3],
                         [0, 4, 5]],
                        [[1, 2, 0],
                         [5, 6, 4],
                         [6, 1, 2],
                         [0, 3, 4]]])
  tf.matrix_set_diag(input, diagonals, k = (-1, 2))
    ==> [[[1, 6, 9, 7],  # Output shape: (2, 3, 4)
          [4, 2, 5, 1],
          [7, 5, 3, 8]],
         [[6, 5, 1, 7],
          [3, 1, 6, 2],
          [7, 4, 2, 4]]]

  # RIGHT_LEFT alignment.
  diagonals = np.array([[[0, 9, 1],  # Diagonal shape: (2, 4, 3)
                         [6, 5, 8],
                         [1, 2, 3],
                         [4, 5, 0]],
                        [[0, 1, 2],
                         [5, 6, 4],
                         [6, 1, 2],
                         [3, 4, 0]]])
  tf.matrix_set_diag(input, diagonals, k = (-1, 2), align="RIGHT_LEFT")
    ==> [[[1, 6, 9, 7],  # Output shape: (2, 3, 4)
          [4, 2, 5, 1],
          [7, 5, 3, 8]],
         [[6, 5, 1, 7],
          [3, 1, 6, 2],
          [7, 4, 2, 4]]]

  ```

  Args:
    input: A `Tensor` with rank `k + 1`, where `k >= 1`.
    diagonal:  A `Tensor` with rank `k`, when `d_lower == d_upper`, or `k + 1`,
      otherwise. `k >= 1`.
    name: A name for the operation (optional).
    k: Diagonal offset(s). Positive value means superdiagonal, 0 refers to the
      main diagonal, and negative value means subdiagonals. `k` can be a single
      integer (for a single diagonal) or a pair of integers specifying the low
      and high ends of a matrix band. `k[0]` must not be larger than `k[1]`.
    align: Some diagonals are shorter than `max_diag_len` and need to be padded.
      `align` is a string specifying how superdiagonals and subdiagonals should
      be aligned, respectively. There are four possible alignments: "RIGHT_LEFT"
      (default), "LEFT_RIGHT", "LEFT_LEFT", and "RIGHT_RIGHT". "RIGHT_LEFT"
      aligns superdiagonals to the right (left-pads the row) and subdiagonals to
      the left (right-pads the row). It is the packing format LAPACK uses.
      cuSPARSE uses "LEFT_RIGHT", which is the opposite alignment.
  """
    return gen_array_ops.matrix_set_diag_v3(input=input, diagonal=diagonal, k=k, align=align, name=name)