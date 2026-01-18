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
def image_summary_eager_fallback(tag: _atypes.TensorFuzzingAnnotation[_atypes.String], tensor: _atypes.TensorFuzzingAnnotation[TV_ImageSummary_T], max_images: int, bad_color, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    if max_images is None:
        max_images = 3
    max_images = _execute.make_int(max_images, 'max_images')
    if bad_color is None:
        bad_color = _execute.make_tensor('dtype: DT_UINT8 tensor_shape { dim { size: 4 } } int_val: 255 int_val: 0 int_val: 0 int_val: 255 ', 'bad_color')
    bad_color = _execute.make_tensor(bad_color, 'bad_color')
    _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [_dtypes.uint8, _dtypes.float32, _dtypes.half, _dtypes.float64], _dtypes.float32)
    tag = _ops.convert_to_tensor(tag, _dtypes.string)
    _inputs_flat = [tag, tensor]
    _attrs = ('max_images', max_images, 'T', _attr_T, 'bad_color', bad_color)
    _result = _execute.execute(b'ImageSummary', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ImageSummary', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result