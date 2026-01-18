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
def resize_area(images: _atypes.TensorFuzzingAnnotation[TV_ResizeArea_T], size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], align_corners: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Resize `images` to `size` using area interpolation.

  Input images can be of different types but output images are always float.

  The range of pixel values for the output image might be slightly different
  from the range for the input image because of limited numerical precision.
  To guarantee an output range, for example `[0.0, 1.0]`, apply
  `tf.clip_by_value` to the output.

  Each output pixel is computed by first transforming the pixel's footprint into
  the input tensor and then averaging the pixels that intersect the footprint. An
  input pixel's contribution to the average is weighted by the fraction of its
  area that intersects the footprint.  This is the same as OpenCV's INTER_AREA.

  Args:
    images: A `Tensor`. Must be one of the following types: `int8`, `uint8`, `int16`, `uint16`, `int32`, `int64`, `half`, `float32`, `float64`, `bfloat16`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and output tensors are
      aligned, preserving the values at the corner pixels. Defaults to false.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResizeArea', name, images, size, 'align_corners', align_corners)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resize_area_eager_fallback(images, size, align_corners=align_corners, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if align_corners is None:
        align_corners = False
    align_corners = _execute.make_bool(align_corners, 'align_corners')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResizeArea', images=images, size=size, align_corners=align_corners, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'align_corners', _op._get_attr_bool('align_corners'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ResizeArea', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result