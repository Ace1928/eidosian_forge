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
def batch_matrix_solve_ls(matrix: _atypes.TensorFuzzingAnnotation[TV_BatchMatrixSolveLs_T], rhs: _atypes.TensorFuzzingAnnotation[TV_BatchMatrixSolveLs_T], l2_regularizer: _atypes.TensorFuzzingAnnotation[_atypes.Float64], fast: bool=True, name=None) -> _atypes.TensorFuzzingAnnotation[TV_BatchMatrixSolveLs_T]:
    """TODO: add doc.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
    l2_regularizer: A `Tensor` of type `float64`.
    fast: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BatchMatrixSolveLs', name, matrix, rhs, l2_regularizer, 'fast', fast)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return batch_matrix_solve_ls_eager_fallback(matrix, rhs, l2_regularizer, fast=fast, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if fast is None:
        fast = True
    fast = _execute.make_bool(fast, 'fast')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BatchMatrixSolveLs', matrix=matrix, rhs=rhs, l2_regularizer=l2_regularizer, fast=fast, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'fast', _op._get_attr_bool('fast'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BatchMatrixSolveLs', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result