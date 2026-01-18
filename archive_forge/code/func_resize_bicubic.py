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
def resize_bicubic(images: _atypes.TensorFuzzingAnnotation[TV_ResizeBicubic_T], size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], align_corners: bool=False, half_pixel_centers: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Resize `images` to `size` using bicubic interpolation.

  Input images can be of different types but output images are always float.

  Args:
    images: A `Tensor`. Must be one of the following types: `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`, `half`, `float32`, `float64`, `bfloat16`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and output tensors are
      aligned, preserving the values at the corner pixels. Defaults to false.
    half_pixel_centers: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResizeBicubic', name, images, size, 'align_corners', align_corners, 'half_pixel_centers', half_pixel_centers)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resize_bicubic_eager_fallback(images, size, align_corners=align_corners, half_pixel_centers=half_pixel_centers, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if align_corners is None:
        align_corners = False
    align_corners = _execute.make_bool(align_corners, 'align_corners')
    if half_pixel_centers is None:
        half_pixel_centers = False
    half_pixel_centers = _execute.make_bool(half_pixel_centers, 'half_pixel_centers')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResizeBicubic', images=images, size=size, align_corners=align_corners, half_pixel_centers=half_pixel_centers, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'align_corners', _op._get_attr_bool('align_corners'), 'half_pixel_centers', _op._get_attr_bool('half_pixel_centers'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ResizeBicubic', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result