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
def prevent_gradient(input: _atypes.TensorFuzzingAnnotation[TV_PreventGradient_T], message: str='', name=None) -> _atypes.TensorFuzzingAnnotation[TV_PreventGradient_T]:
    """An identity op that triggers an error if a gradient is requested.

  When executed in a graph, this op outputs its input tensor as-is.

  When building ops to compute gradients, the TensorFlow gradient system
  will return an error when trying to lookup the gradient of this op,
  because no gradient must ever be registered for this function.  This
  op exists to prevent subtle bugs from silently returning unimplemented
  gradients in some corner cases.

  Args:
    input: A `Tensor`. any tensor.
    message: An optional `string`. Defaults to `""`.
      Will be printed in the error when anyone tries to differentiate
      this operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'PreventGradient', name, input, 'message', message)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return prevent_gradient_eager_fallback(input, message=message, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if message is None:
        message = ''
    message = _execute.make_str(message, 'message')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('PreventGradient', input=input, message=message, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'message', _op.get_attr('message'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('PreventGradient', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result