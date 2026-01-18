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
def tensor_list_set_item_eager_fallback(input_handle: _atypes.TensorFuzzingAnnotation[_atypes.Variant], index: _atypes.TensorFuzzingAnnotation[_atypes.Int32], item: _atypes.TensorFuzzingAnnotation[TV_TensorListSetItem_element_dtype], resize_if_index_out_of_bounds: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if resize_if_index_out_of_bounds is None:
        resize_if_index_out_of_bounds = False
    resize_if_index_out_of_bounds = _execute.make_bool(resize_if_index_out_of_bounds, 'resize_if_index_out_of_bounds')
    _attr_element_dtype, (item,) = _execute.args_to_matching_eager([item], ctx, [])
    input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
    index = _ops.convert_to_tensor(index, _dtypes.int32)
    _inputs_flat = [input_handle, index, item]
    _attrs = ('element_dtype', _attr_element_dtype, 'resize_if_index_out_of_bounds', resize_if_index_out_of_bounds)
    _result = _execute.execute(b'TensorListSetItem', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TensorListSetItem', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result