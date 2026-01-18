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
def debug_identity(input: _atypes.TensorFuzzingAnnotation[TV_DebugIdentity_T], device_name: str='', tensor_name: str='', debug_urls=[], gated_grpc: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_DebugIdentity_T]:
    """Provides an identity mapping of the non-Ref type input tensor for debugging.

  Provides an identity mapping of the non-Ref type input tensor for debugging.

  Args:
    input: A `Tensor`. Input tensor, non-Reference type
    device_name: An optional `string`. Defaults to `""`.
      Name of the device on which the tensor resides.
    tensor_name: An optional `string`. Defaults to `""`.
      Name of the input tensor.
    debug_urls: An optional list of `strings`. Defaults to `[]`.
      List of URLs to debug targets, e.g.,
        file:///foo/tfdbg_dump, grpc:://localhost:11011
    gated_grpc: An optional `bool`. Defaults to `False`.
      Whether this op will be gated. If any of the debug_urls of this
        debug node is of the grpc:// scheme, when the value of this attribute is set
        to True, the data will not actually be sent via the grpc stream unless this
        debug op has been enabled at the debug_url. If all of the debug_urls of this
        debug node are of the grpc:// scheme and the debug op is enabled at none of
        them, the output will be an empty Tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DebugIdentity', name, input, 'device_name', device_name, 'tensor_name', tensor_name, 'debug_urls', debug_urls, 'gated_grpc', gated_grpc)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return debug_identity_eager_fallback(input, device_name=device_name, tensor_name=tensor_name, debug_urls=debug_urls, gated_grpc=gated_grpc, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if device_name is None:
        device_name = ''
    device_name = _execute.make_str(device_name, 'device_name')
    if tensor_name is None:
        tensor_name = ''
    tensor_name = _execute.make_str(tensor_name, 'tensor_name')
    if debug_urls is None:
        debug_urls = []
    if not isinstance(debug_urls, (list, tuple)):
        raise TypeError("Expected list for 'debug_urls' argument to 'debug_identity' Op, not %r." % debug_urls)
    debug_urls = [_execute.make_str(_s, 'debug_urls') for _s in debug_urls]
    if gated_grpc is None:
        gated_grpc = False
    gated_grpc = _execute.make_bool(gated_grpc, 'gated_grpc')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DebugIdentity', input=input, device_name=device_name, tensor_name=tensor_name, debug_urls=debug_urls, gated_grpc=gated_grpc, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'device_name', _op.get_attr('device_name'), 'tensor_name', _op.get_attr('tensor_name'), 'debug_urls', _op.get_attr('debug_urls'), 'gated_grpc', _op._get_attr_bool('gated_grpc'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DebugIdentity', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result