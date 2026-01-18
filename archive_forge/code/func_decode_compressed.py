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
@tf_export('io.decode_compressed', v1=['io.decode_compressed', 'decode_compressed'])
@deprecated_endpoints('decode_compressed')
def decode_compressed(bytes: _atypes.TensorFuzzingAnnotation[_atypes.String], compression_type: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Decompress strings.

  This op decompresses each element of the `bytes` input `Tensor`, which
  is assumed to be compressed using the given `compression_type`.

  The `output` is a string `Tensor` of the same shape as `bytes`,
  each element containing the decompressed data from the corresponding
  element in `bytes`.

  Args:
    bytes: A `Tensor` of type `string`.
      A Tensor of string which is compressed.
    compression_type: An optional `string`. Defaults to `""`.
      A scalar containing either (i) the empty string (no
      compression), (ii) "ZLIB", or (iii) "GZIP".
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DecodeCompressed', name, bytes, 'compression_type', compression_type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_decode_compressed((bytes, compression_type, name), None)
            if _result is not NotImplemented:
                return _result
            return decode_compressed_eager_fallback(bytes, compression_type=compression_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(decode_compressed, (), dict(bytes=bytes, compression_type=compression_type, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_decode_compressed((bytes, compression_type, name), None)
        if _result is not NotImplemented:
            return _result
    if compression_type is None:
        compression_type = ''
    compression_type = _execute.make_str(compression_type, 'compression_type')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('DecodeCompressed', bytes=bytes, compression_type=compression_type, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(decode_compressed, (), dict(bytes=bytes, compression_type=compression_type, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('compression_type', _op.get_attr('compression_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DecodeCompressed', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result