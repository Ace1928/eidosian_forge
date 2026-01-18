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
@tf_export('io.encode_base64', v1=['io.encode_base64', 'encode_base64'])
@deprecated_endpoints('encode_base64')
def encode_base64(input: _atypes.TensorFuzzingAnnotation[_atypes.String], pad: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Encode strings into web-safe base64 format.

  Refer to [this article](https://en.wikipedia.org/wiki/Base64) for more information on
  base64 format. Base64 strings may have padding with '=' at the
  end so that the encoded has length multiple of 4. See Padding section of the
  link above.

  Web-safe means that the encoder uses - and _ instead of + and /.

  Args:
    input: A `Tensor` of type `string`. Strings to be encoded.
    pad: An optional `bool`. Defaults to `False`.
      Bool whether padding is applied at the ends.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'EncodeBase64', name, input, 'pad', pad)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_encode_base64((input, pad, name), None)
            if _result is not NotImplemented:
                return _result
            return encode_base64_eager_fallback(input, pad=pad, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(encode_base64, (), dict(input=input, pad=pad, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_encode_base64((input, pad, name), None)
        if _result is not NotImplemented:
            return _result
    if pad is None:
        pad = False
    pad = _execute.make_bool(pad, 'pad')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('EncodeBase64', input=input, pad=pad, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(encode_base64, (), dict(input=input, pad=pad, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('pad', _op._get_attr_bool('pad'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('EncodeBase64', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result