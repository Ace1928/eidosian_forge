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
def sparse_sparse_minimum(a_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], a_values: _atypes.TensorFuzzingAnnotation[TV_SparseSparseMinimum_T], a_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], b_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], b_values: _atypes.TensorFuzzingAnnotation[TV_SparseSparseMinimum_T], b_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name=None):
    """Returns the element-wise min of two SparseTensors.

  Assumes the two SparseTensors have the same shape, i.e., no broadcasting.

  Args:
    a_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, in the canonical lexicographic ordering.
    a_values: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      1-D.  `N` non-empty values corresponding to `a_indices`.
    a_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    b_indices: A `Tensor` of type `int64`.
      counterpart to `a_indices` for the other operand.
    b_values: A `Tensor`. Must have the same type as `a_values`.
      counterpart to `a_values` for the other operand; must be of the same dtype.
    b_shape: A `Tensor` of type `int64`.
      counterpart to `a_shape` for the other operand; the two shapes must be equal.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values).

    output_indices: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `a_values`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseSparseMinimum', name, a_indices, a_values, a_shape, b_indices, b_values, b_shape)
            _result = _SparseSparseMinimumOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_sparse_minimum_eager_fallback(a_indices, a_values, a_shape, b_indices, b_values, b_shape, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseSparseMinimum', a_indices=a_indices, a_values=a_values, a_shape=a_shape, b_indices=b_indices, b_values=b_values, b_shape=b_shape, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseSparseMinimum', _inputs_flat, _attrs, _result)
    _result = _SparseSparseMinimumOutput._make(_result)
    return _result