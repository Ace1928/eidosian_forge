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
def adjust_contrast_eager_fallback(images: _atypes.TensorFuzzingAnnotation[TV_AdjustContrast_T], contrast_factor: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_value: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_value: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.uint8, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.float32, _dtypes.float64])
    contrast_factor = _ops.convert_to_tensor(contrast_factor, _dtypes.float32)
    min_value = _ops.convert_to_tensor(min_value, _dtypes.float32)
    max_value = _ops.convert_to_tensor(max_value, _dtypes.float32)
    _inputs_flat = [images, contrast_factor, min_value, max_value]
    _attrs = ('T', _attr_T)
    _result = _execute.execute(b'AdjustContrast', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('AdjustContrast', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result