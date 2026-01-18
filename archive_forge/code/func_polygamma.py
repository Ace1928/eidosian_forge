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
@tf_export('math.polygamma', v1=['math.polygamma', 'polygamma'])
@deprecated_endpoints('polygamma')
def polygamma(a: _atypes.TensorFuzzingAnnotation[TV_Polygamma_T], x: _atypes.TensorFuzzingAnnotation[TV_Polygamma_T], name=None) -> _atypes.TensorFuzzingAnnotation[TV_Polygamma_T]:
    """Compute the polygamma function \\\\(\\psi^{(n)}(x)\\\\).

  The polygamma function is defined as:


  \\\\(\\psi^{(a)}(x) = \\frac{d^a}{dx^a} \\psi(x)\\\\)

  where \\\\(\\psi(x)\\\\) is the digamma function.
  The polygamma function is defined only for non-negative integer orders \\\\a\\\\.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    x: A `Tensor`. Must have the same type as `a`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Polygamma', name, a, x)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_polygamma((a, x, name), None)
            if _result is not NotImplemented:
                return _result
            return polygamma_eager_fallback(a, x, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(polygamma, (), dict(a=a, x=x, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_polygamma((a, x, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('Polygamma', a=a, x=x, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(polygamma, (), dict(a=a, x=x, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Polygamma', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result