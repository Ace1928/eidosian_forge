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
@tf_export('io.parse_tensor', v1=['io.parse_tensor', 'parse_tensor'])
@deprecated_endpoints('parse_tensor')
def parse_tensor(serialized: _atypes.TensorFuzzingAnnotation[_atypes.String], out_type: TV_ParseTensor_out_type, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ParseTensor_out_type]:
    """Transforms a serialized tensorflow.TensorProto proto into a Tensor.

  Args:
    serialized: A `Tensor` of type `string`.
      A scalar string containing a serialized TensorProto proto.
    out_type: A `tf.DType`.
      The type of the serialized tensor.  The provided type must match the
      type of the serialized tensor and no implicit conversion will take place.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `out_type`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ParseTensor', name, serialized, 'out_type', out_type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_parse_tensor((serialized, out_type, name), None)
            if _result is not NotImplemented:
                return _result
            return parse_tensor_eager_fallback(serialized, out_type=out_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(parse_tensor, (), dict(serialized=serialized, out_type=out_type, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_parse_tensor((serialized, out_type, name), None)
        if _result is not NotImplemented:
            return _result
    out_type = _execute.make_type(out_type, 'out_type')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('ParseTensor', serialized=serialized, out_type=out_type, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(parse_tensor, (), dict(serialized=serialized, out_type=out_type, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('out_type', _op._get_attr_type('out_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ParseTensor', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result