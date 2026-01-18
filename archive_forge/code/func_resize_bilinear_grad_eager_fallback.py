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
def resize_bilinear_grad_eager_fallback(grads: _atypes.TensorFuzzingAnnotation[_atypes.Float32], original_image: _atypes.TensorFuzzingAnnotation[TV_ResizeBilinearGrad_T], align_corners: bool, half_pixel_centers: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_ResizeBilinearGrad_T]:
    if align_corners is None:
        align_corners = False
    align_corners = _execute.make_bool(align_corners, 'align_corners')
    if half_pixel_centers is None:
        half_pixel_centers = False
    half_pixel_centers = _execute.make_bool(half_pixel_centers, 'half_pixel_centers')
    _attr_T, (original_image,) = _execute.args_to_matching_eager([original_image], ctx, [_dtypes.float32, _dtypes.bfloat16, _dtypes.half, _dtypes.float64])
    grads = _ops.convert_to_tensor(grads, _dtypes.float32)
    _inputs_flat = [grads, original_image]
    _attrs = ('T', _attr_T, 'align_corners', align_corners, 'half_pixel_centers', half_pixel_centers)
    _result = _execute.execute(b'ResizeBilinearGrad', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ResizeBilinearGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result