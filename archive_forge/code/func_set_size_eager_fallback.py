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
def set_size_eager_fallback(set_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], set_values: _atypes.TensorFuzzingAnnotation[TV_SetSize_T], set_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], validate_indices: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Int32]:
    if validate_indices is None:
        validate_indices = True
    validate_indices = _execute.make_bool(validate_indices, 'validate_indices')
    _attr_T, (set_values,) = _execute.args_to_matching_eager([set_values], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.string])
    set_indices = _ops.convert_to_tensor(set_indices, _dtypes.int64)
    set_shape = _ops.convert_to_tensor(set_shape, _dtypes.int64)
    _inputs_flat = [set_indices, set_values, set_shape]
    _attrs = ('validate_indices', validate_indices, 'T', _attr_T)
    _result = _execute.execute(b'SetSize', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SetSize', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result