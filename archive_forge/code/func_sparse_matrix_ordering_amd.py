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
def sparse_matrix_ordering_amd(input: _atypes.TensorFuzzingAnnotation[_atypes.Variant], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int32]:
    """Computes the Approximate Minimum Degree (AMD) ordering of `input`.

  Computes the Approximate Minimum Degree (AMD) ordering for a sparse matrix.

  The returned permutation may be used to permute the rows and columns of the
  given sparse matrix. This typically results in permuted sparse matrix's sparse
  Cholesky (or other decompositions) in having fewer zero fill-in compared to
  decomposition of the original matrix.

  The input sparse matrix may have rank 2 or rank 3. The output Tensor,
  representing would then have rank 1 or 2 respectively, with the same batch
  shape as the input.

  Each component of the input sparse matrix must represent a square symmetric
  matrix; only the lower triangular part of the matrix is read. The values of the
  sparse matrix does not affect the returned permutation, only the sparsity
  pattern of the sparse matrix is used. Hence, a single AMD ordering may be
  reused for the Cholesky decompositions of sparse matrices with the same sparsity
  pattern but with possibly different values.

  Each batch component of the output permutation represents a permutation of `N`
  elements, where the input sparse matrix components each have `N` rows. That is,
  the component contains each of the integers `{0, .. N-1}` exactly once. The
  `i`th element represents the row index that the `i`th row maps to.

  Usage example:

  ```python
      from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops

      a_indices = np.array([[0, 0], [1, 1], [2, 1], [2, 2], [3, 3]])
      a_values = np.array([1.0, 2.0, 1.0, 3.0, 4.0], np.float32)
      a_dense_shape = [4, 4]

      with tf.Session() as sess:
        # Define (COO format) SparseTensor over Numpy array.
        a_st = tf.sparse.SparseTensor(a_indices, a_values, a_dense_shape)

        # Convert SparseTensors to CSR SparseMatrix.
        a_sm = sparse_csr_matrix_ops.sparse_tensor_to_csr_sparse_matrix(
            a_st.indices, a_st.values, a_st.dense_shape)

        # Obtain the AMD Ordering for the CSR SparseMatrix.
        ordering_amd = sparse_csr_matrix_ops.sparse_matrix_ordering_amd(sparse_matrix)

        ordering_amd_value = sess.run(ordering_amd)
  ```

  `ordering_amd_value` stores the AMD ordering: `[1 2 3 0]`.

  input: A `CSRSparseMatrix`.

  Args:
    input: A `Tensor` of type `variant`. A `CSRSparseMatrix`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseMatrixOrderingAMD', name, input)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_matrix_ordering_amd_eager_fallback(input, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseMatrixOrderingAMD', input=input, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseMatrixOrderingAMD', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result