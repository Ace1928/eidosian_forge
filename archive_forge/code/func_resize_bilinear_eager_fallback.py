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
def resize_bilinear_eager_fallback(images: _atypes.TensorFuzzingAnnotation[TV_ResizeBilinear_T], size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], align_corners: bool, half_pixel_centers: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    if align_corners is None:
        align_corners = False
    align_corners = _execute.make_bool(align_corners, 'align_corners')
    if half_pixel_centers is None:
        half_pixel_centers = False
    half_pixel_centers = _execute.make_bool(half_pixel_centers, 'half_pixel_centers')
    _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.int8, _dtypes.uint8, _dtypes.int16, _dtypes.uint16, _dtypes.int32, _dtypes.int64, _dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.bfloat16])
    size = _ops.convert_to_tensor(size, _dtypes.int32)
    _inputs_flat = [images, size]
    _attrs = ('T', _attr_T, 'align_corners', align_corners, 'half_pixel_centers', half_pixel_centers)
    _result = _execute.execute(b'ResizeBilinear', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ResizeBilinear', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result