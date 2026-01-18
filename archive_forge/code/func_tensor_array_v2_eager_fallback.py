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
def tensor_array_v2_eager_fallback(size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], dtype: TV_TensorArrayV2_dtype, element_shape, dynamic_size: bool, clear_after_read: bool, tensor_array_name: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    dtype = _execute.make_type(dtype, 'dtype')
    if element_shape is None:
        element_shape = None
    element_shape = _execute.make_shape(element_shape, 'element_shape')
    if dynamic_size is None:
        dynamic_size = False
    dynamic_size = _execute.make_bool(dynamic_size, 'dynamic_size')
    if clear_after_read is None:
        clear_after_read = True
    clear_after_read = _execute.make_bool(clear_after_read, 'clear_after_read')
    if tensor_array_name is None:
        tensor_array_name = ''
    tensor_array_name = _execute.make_str(tensor_array_name, 'tensor_array_name')
    size = _ops.convert_to_tensor(size, _dtypes.int32)
    _inputs_flat = [size]
    _attrs = ('dtype', dtype, 'element_shape', element_shape, 'dynamic_size', dynamic_size, 'clear_after_read', clear_after_read, 'tensor_array_name', tensor_array_name)
    _result = _execute.execute(b'TensorArrayV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TensorArrayV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result