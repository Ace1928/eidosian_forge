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
def approximate_equal(x: _atypes.TensorFuzzingAnnotation[TV_ApproximateEqual_T], y: _atypes.TensorFuzzingAnnotation[TV_ApproximateEqual_T], tolerance: float=1e-05, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Bool]:
    """Returns the truth value of abs(x-y) < tolerance element-wise.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
    y: A `Tensor`. Must have the same type as `x`.
    tolerance: An optional `float`. Defaults to `1e-05`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ApproximateEqual', name, x, y, 'tolerance', tolerance)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return approximate_equal_eager_fallback(x, y, tolerance=tolerance, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if tolerance is None:
        tolerance = 1e-05
    tolerance = _execute.make_float(tolerance, 'tolerance')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ApproximateEqual', x=x, y=y, tolerance=tolerance, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'tolerance', _op.get_attr('tolerance'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ApproximateEqual', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result