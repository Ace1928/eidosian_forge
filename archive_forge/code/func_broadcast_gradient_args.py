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
def broadcast_gradient_args(s0: _atypes.TensorFuzzingAnnotation[TV_BroadcastGradientArgs_T], s1: _atypes.TensorFuzzingAnnotation[TV_BroadcastGradientArgs_T], name=None):
    """Return the reduction indices for computing gradients of s0 op s1 with broadcast.

  This is typically used by gradient computations for a broadcasting operation.

  Args:
    s0: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    s1: A `Tensor`. Must have the same type as `s0`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (r0, r1).

    r0: A `Tensor`. Has the same type as `s0`.
    r1: A `Tensor`. Has the same type as `s0`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BroadcastGradientArgs', name, s0, s1)
            _result = _BroadcastGradientArgsOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return broadcast_gradient_args_eager_fallback(s0, s1, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BroadcastGradientArgs', s0=s0, s1=s1, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BroadcastGradientArgs', _inputs_flat, _attrs, _result)
    _result = _BroadcastGradientArgsOutput._make(_result)
    return _result