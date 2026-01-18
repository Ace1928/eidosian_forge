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
def csr_sparse_matrix_components(csr_sparse_matrix: _atypes.TensorFuzzingAnnotation[_atypes.Variant], index: _atypes.TensorFuzzingAnnotation[_atypes.Int32], type: TV_CSRSparseMatrixComponents_type, name=None):
    """Reads out the CSR components at batch `index`.

  This op is meant only for debugging / testing, and its interface is not expected
  to be stable.

  Args:
    csr_sparse_matrix: A `Tensor` of type `variant`.
      A batched CSRSparseMatrix.
    index: A `Tensor` of type `int32`.
      The index in `csr_sparse_matrix`'s batch.
    type: A `tf.DType` from: `tf.float32, tf.float64, tf.complex64, tf.complex128`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (row_ptrs, col_inds, values).

    row_ptrs: A `Tensor` of type `int32`.
    col_inds: A `Tensor` of type `int32`.
    values: A `Tensor` of type `type`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CSRSparseMatrixComponents', name, csr_sparse_matrix, index, 'type', type)
            _result = _CSRSparseMatrixComponentsOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return csr_sparse_matrix_components_eager_fallback(csr_sparse_matrix, index, type=type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    type = _execute.make_type(type, 'type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CSRSparseMatrixComponents', csr_sparse_matrix=csr_sparse_matrix, index=index, type=type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('type', _op._get_attr_type('type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CSRSparseMatrixComponents', _inputs_flat, _attrs, _result)
    _result = _CSRSparseMatrixComponentsOutput._make(_result)
    return _result