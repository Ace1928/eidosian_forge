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
def mul_no_nan(x: _atypes.TensorFuzzingAnnotation[TV_MulNoNan_T], y: _atypes.TensorFuzzingAnnotation[TV_MulNoNan_T], name=None) -> _atypes.TensorFuzzingAnnotation[TV_MulNoNan_T]:
    """Returns x * y element-wise. Returns zero if y is zero, even if x if infinite or NaN.

  *NOTE*: `MulNoNan` supports broadcasting. More about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `complex64`, `complex128`.
    y: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MulNoNan', name, x, y)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return mul_no_nan_eager_fallback(x, y, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MulNoNan', x=x, y=y, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MulNoNan', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result