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
@tf_export('tensor_map_insert')
def tensor_map_insert(input_handle: _atypes.TensorFuzzingAnnotation[_atypes.Variant], key: _atypes.TensorFuzzingAnnotation[TV_TensorMapInsert_key_dtype], value: _atypes.TensorFuzzingAnnotation[TV_TensorMapInsert_value_dtype], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Returns a map that is the 'input_handle' with the given key-value pair inserted.

  input_handle: the original map
  output_handle: the map with key and value inserted
  key: the key to be inserted
  value: the value to be inserted

  Args:
    input_handle: A `Tensor` of type `variant`.
    key: A `Tensor`.
    value: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorMapInsert', name, input_handle, key, value)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_tensor_map_insert((input_handle, key, value, name), None)
            if _result is not NotImplemented:
                return _result
            return tensor_map_insert_eager_fallback(input_handle, key, value, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(tensor_map_insert, (), dict(input_handle=input_handle, key=key, value=value, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_tensor_map_insert((input_handle, key, value, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorMapInsert', input_handle=input_handle, key=key, value=value, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(tensor_map_insert, (), dict(input_handle=input_handle, key=key, value=value, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('key_dtype', _op._get_attr_type('key_dtype'), 'value_dtype', _op._get_attr_type('value_dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorMapInsert', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result