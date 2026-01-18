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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('debugging.check_numerics', v1=['debugging.check_numerics', 'check_numerics'])
@deprecated_endpoints('check_numerics')
def check_numerics(tensor: _atypes.TensorFuzzingAnnotation[TV_CheckNumerics_T], message: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_CheckNumerics_T]:
    """Checks a tensor for NaN and Inf values.

  When run, reports an `InvalidArgument` error if `tensor` has any values
  that are not a number (NaN) or infinity (Inf). Otherwise, returns the input
  tensor.

  Example usage:

  ``` python
  a = tf.Variable(1.0)
  tf.debugging.check_numerics(a, message='')

  b = tf.Variable(np.nan)
  try:
    tf.debugging.check_numerics(b, message='Checking b')
  except Exception as e:
    assert "Checking b : Tensor had NaN values" in e.message

  c = tf.Variable(np.inf)
  try:
    tf.debugging.check_numerics(c, message='Checking c')
  except Exception as e:
    assert "Checking c : Tensor had Inf values" in e.message
  ```

  Args:
    tensor: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    message: A `string`. Prefix of the error message.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `tensor`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CheckNumerics', name, tensor, 'message', message)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_check_numerics((tensor, message, name), None)
            if _result is not NotImplemented:
                return _result
            return check_numerics_eager_fallback(tensor, message=message, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(check_numerics, (), dict(tensor=tensor, message=message, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_check_numerics((tensor, message, name), None)
        if _result is not NotImplemented:
            return _result
    message = _execute.make_str(message, 'message')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('CheckNumerics', tensor=tensor, message=message, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(check_numerics, (), dict(tensor=tensor, message=message, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'message', _op.get_attr('message'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CheckNumerics', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result