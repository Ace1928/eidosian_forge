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
def sparse_matrix_zeros(dense_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], type: TV_SparseMatrixZeros_type, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates an all-zeros CSRSparseMatrix with shape `dense_shape`.

  Args:
    dense_shape: A `Tensor` of type `int64`. The desired matrix shape.
    type: A `tf.DType` from: `tf.float32, tf.float64, tf.complex64, tf.complex128`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseMatrixZeros', name, dense_shape, 'type', type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_matrix_zeros_eager_fallback(dense_shape, type=type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    type = _execute.make_type(type, 'type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseMatrixZeros', dense_shape=dense_shape, type=type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('type', _op._get_attr_type('type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseMatrixZeros', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result