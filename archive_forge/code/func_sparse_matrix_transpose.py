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
def sparse_matrix_transpose(input: _atypes.TensorFuzzingAnnotation[_atypes.Variant], type: TV_SparseMatrixTranspose_type, conjugate: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Transposes the inner (matrix) dimensions of a CSRSparseMatrix.

  Transposes the inner (matrix) dimensions of a SparseMatrix and optionally
  conjugates its values.

  Args:
    input: A `Tensor` of type `variant`. A CSRSparseMatrix.
    type: A `tf.DType` from: `tf.float32, tf.float64, tf.complex64, tf.complex128`.
    conjugate: An optional `bool`. Defaults to `False`.
      Indicates whether `input` should be conjugated.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseMatrixTranspose', name, input, 'conjugate', conjugate, 'type', type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_matrix_transpose_eager_fallback(input, conjugate=conjugate, type=type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    type = _execute.make_type(type, 'type')
    if conjugate is None:
        conjugate = False
    conjugate = _execute.make_bool(conjugate, 'conjugate')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseMatrixTranspose', input=input, type=type, conjugate=conjugate, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('conjugate', _op._get_attr_bool('conjugate'), 'type', _op._get_attr_type('type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseMatrixTranspose', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result