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
@tf_export('rpc_call')
def rpc_call(client: _atypes.TensorFuzzingAnnotation[_atypes.Resource], method_name: _atypes.TensorFuzzingAnnotation[_atypes.String], args, timeout_in_ms: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name=None):
    """TODO: add doc.

  Args:
    client: A `Tensor` of type `resource`.
    method_name: A `Tensor` of type `string`.
    args: A list of `Tensor` objects.
    timeout_in_ms: A `Tensor` of type `int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (future, deleter).

    future: A `Tensor` of type `resource`.
    deleter: A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RpcCall', name, client, method_name, args, timeout_in_ms)
            _result = _RpcCallOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_rpc_call((client, method_name, args, timeout_in_ms, name), None)
            if _result is not NotImplemented:
                return _result
            return rpc_call_eager_fallback(client, method_name, args, timeout_in_ms, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(rpc_call, (), dict(client=client, method_name=method_name, args=args, timeout_in_ms=timeout_in_ms, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_rpc_call((client, method_name, args, timeout_in_ms, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('RpcCall', client=client, method_name=method_name, args=args, timeout_in_ms=timeout_in_ms, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(rpc_call, (), dict(client=client, method_name=method_name, args=args, timeout_in_ms=timeout_in_ms, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tin', _op.get_attr('Tin'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RpcCall', _inputs_flat, _attrs, _result)
    _result = _RpcCallOutput._make(_result)
    return _result