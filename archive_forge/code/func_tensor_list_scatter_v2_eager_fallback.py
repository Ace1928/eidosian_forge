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
def tensor_list_scatter_v2_eager_fallback(tensor: _atypes.TensorFuzzingAnnotation[TV_TensorListScatterV2_element_dtype], indices: _atypes.TensorFuzzingAnnotation[_atypes.Int32], element_shape: _atypes.TensorFuzzingAnnotation[TV_TensorListScatterV2_shape_type], num_elements: _atypes.TensorFuzzingAnnotation[_atypes.Int32], name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    _attr_element_dtype, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
    _attr_shape_type, (element_shape,) = _execute.args_to_matching_eager([element_shape], ctx, [_dtypes.int32, _dtypes.int64])
    indices = _ops.convert_to_tensor(indices, _dtypes.int32)
    num_elements = _ops.convert_to_tensor(num_elements, _dtypes.int32)
    _inputs_flat = [tensor, indices, element_shape, num_elements]
    _attrs = ('element_dtype', _attr_element_dtype, 'shape_type', _attr_shape_type)
    _result = _execute.execute(b'TensorListScatterV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TensorListScatterV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result