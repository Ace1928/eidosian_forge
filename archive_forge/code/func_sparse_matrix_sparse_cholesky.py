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
def sparse_matrix_sparse_cholesky(input: _atypes.TensorFuzzingAnnotation[_atypes.Variant], permutation: _atypes.TensorFuzzingAnnotation[_atypes.Int32], type: TV_SparseMatrixSparseCholesky_type, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Computes the sparse Cholesky decomposition of `input`.

  Computes the Sparse Cholesky decomposition of a sparse matrix, with the given
  fill-in reducing permutation.

  The input sparse matrix and the fill-in reducing permutation `permutation` must
  have compatible shapes. If the sparse matrix has rank 3; with the batch
  dimension `B`, then the `permutation` must be of rank 2; with the same batch
  dimension `B`. There is no support for broadcasting.

  Furthermore, each component vector of `permutation` must be of length `N`,
  containing each of the integers {0, 1, ..., N - 1} exactly once, where `N` is
  the number of rows of each component of the sparse matrix.

  Each component of the input sparse matrix must represent a symmetric positive
  definite (SPD) matrix; although only the lower triangular part of the matrix is
  read. If any individual component is not SPD, then an InvalidArgument error is
  thrown.

  The returned sparse matrix has the same dense shape as the input sparse matrix.
  For each component `A` of the input sparse matrix, the corresponding output
  sparse matrix represents `L`, the lower triangular Cholesky factor satisfying
  the following identity:

  ```
    A = L * Lt
  ```

  where Lt denotes the transpose of L (or its conjugate transpose, if `type` is
  `complex64` or `complex128`).

  The `type` parameter denotes the type of the matrix elements. The supported
  types are: `float32`, `float64`, `complex64` and `complex128`.

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

        # Obtain the Sparse Cholesky factor using AMD Ordering for reducing zero
        # fill-in (number of structural non-zeros in the sparse Cholesky factor).
        ordering_amd = sparse_csr_matrix_ops.sparse_matrix_ordering_amd(sparse_matrix)
        cholesky_sparse_matrices = (
            sparse_csr_matrix_ops.sparse_matrix_sparse_cholesky(
                sparse_matrix, ordering_amd, type=tf.float32))

        # Convert the CSRSparseMatrix Cholesky factor to a dense Tensor
        dense_cholesky = sparse_csr_matrix_ops.csr_sparse_matrix_to_dense(
            cholesky_sparse_matrices, tf.float32)

        # Evaluate the dense Tensor value.
        dense_cholesky_value = sess.run(dense_cholesky)
  ```

  `dense_cholesky_value` stores the dense Cholesky factor:

  ```
      [[  1.  0.    0.    0.]
       [  0.  1.41  0.    0.]
       [  0.  0.70  1.58  0.]
       [  0.  0.    0.    2.]]
  ```


  input: A `CSRSparseMatrix`.
  permutation: A `Tensor`.
  type: The type of `input`.

  Args:
    input: A `Tensor` of type `variant`. A `CSRSparseMatrix`.
    permutation: A `Tensor` of type `int32`.
      A fill-in reducing permutation matrix.
    type: A `tf.DType` from: `tf.float32, tf.float64, tf.complex64, tf.complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseMatrixSparseCholesky', name, input, permutation, 'type', type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_matrix_sparse_cholesky_eager_fallback(input, permutation, type=type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    type = _execute.make_type(type, 'type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseMatrixSparseCholesky', input=input, permutation=permutation, type=type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('type', _op._get_attr_type('type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseMatrixSparseCholesky', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result