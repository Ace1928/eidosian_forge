import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def matrix_set_diag_v2(input: _atypes.TensorFuzzingAnnotation[TV_MatrixSetDiagV2_T], diagonal: _atypes.TensorFuzzingAnnotation[TV_MatrixSetDiagV2_T], k: _atypes.TensorFuzzingAnnotation[_atypes.Int32], name=None) -> _atypes.TensorFuzzingAnnotation[TV_MatrixSetDiagV2_T]:
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
  where `d = n - m`, `diag_index = k[1] - d`, and `index_in_diag = n - max(d, 0)`.

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
  tf.matrix_set_diag(diagonal) ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
                                     [7, 2, 7, 7],
                                     [7, 7, 3, 7]],
                                    [[4, 7, 7, 7],
                                     [7, 5, 7, 7],
                                     [7, 7, 6, 7]]]

  # A superdiagonal (per batch).
  tf.matrix_set_diag(diagonal, k = 1)
    ==> [[[7, 1, 7, 7],  # Output shape: (2, 3, 4)
          [7, 7, 2, 7],
          [7, 7, 7, 3]],
         [[7, 4, 7, 7],
          [7, 7, 5, 7],
          [7, 7, 7, 6]]]

  # A band of diagonals.
  diagonals = np.array([[[1, 2, 3],  # Diagonal shape: (2, 2, 3)
                         [4, 5, 0]],
                        [[6, 1, 2],
                         [3, 4, 0]]])
  tf.matrix_set_diag(diagonals, k = (-1, 0))
    ==> [[[1, 7, 7, 7],  # Output shape: (2, 3, 4)
          [4, 2, 7, 7],
          [0, 5, 3, 7]],
         [[6, 7, 7, 7],
          [3, 1, 7, 7],
          [7, 4, 2, 7]]]

  ```

  Args:
    input: A `Tensor`. Rank `r+1`, where `r >= 1`.
    diagonal: A `Tensor`. Must have the same type as `input`.
      Rank `r` when `k` is an integer or `k[0] == k[1]`. Otherwise, it has rank `r+1`.
      `k >= 1`.
    k: A `Tensor` of type `int32`.
      Diagonal offset(s). Positive value means superdiagonal, 0 refers to the main
      diagonal, and negative value means subdiagonals. `k` can be a single integer
      (for a single diagonal) or a pair of integers specifying the low and high ends
      of a matrix band. `k[0]` must not be larger than `k[1]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MatrixSetDiagV2', name, input, diagonal, k)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return matrix_set_diag_v2_eager_fallback(input, diagonal, k, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MatrixSetDiagV2', input=input, diagonal=diagonal, k=k, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MatrixSetDiagV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result