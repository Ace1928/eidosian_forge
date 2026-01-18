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
def tensor_map_erase_eager_fallback(input_handle: _atypes.TensorFuzzingAnnotation[_atypes.Variant], key: _atypes.TensorFuzzingAnnotation[TV_TensorMapErase_key_dtype], value_dtype: TV_TensorMapErase_value_dtype, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    value_dtype = _execute.make_type(value_dtype, 'value_dtype')
    _attr_key_dtype, (key,) = _execute.args_to_matching_eager([key], ctx, [])
    input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
    _inputs_flat = [input_handle, key]
    _attrs = ('key_dtype', _attr_key_dtype, 'value_dtype', value_dtype)
    _result = _execute.execute(b'TensorMapErase', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TensorMapErase', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result