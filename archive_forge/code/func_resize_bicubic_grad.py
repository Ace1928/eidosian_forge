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
def resize_bicubic_grad(grads: _atypes.TensorFuzzingAnnotation[_atypes.Float32], original_image: _atypes.TensorFuzzingAnnotation[TV_ResizeBicubicGrad_T], align_corners: bool=False, half_pixel_centers: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ResizeBicubicGrad_T]:
    """Computes the gradient of bicubic interpolation.

  Args:
    grads: A `Tensor` of type `float32`.
      4-D with shape `[batch, height, width, channels]`.
    original_image: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      4-D with shape `[batch, orig_height, orig_width, channels]`,
      The image tensor that was resized.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, the centers of the 4 corner pixels of the input and grad tensors are
      aligned. Defaults to false.
    half_pixel_centers: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `original_image`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResizeBicubicGrad', name, grads, original_image, 'align_corners', align_corners, 'half_pixel_centers', half_pixel_centers)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resize_bicubic_grad_eager_fallback(grads, original_image, align_corners=align_corners, half_pixel_centers=half_pixel_centers, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if align_corners is None:
        align_corners = False
    align_corners = _execute.make_bool(align_corners, 'align_corners')
    if half_pixel_centers is None:
        half_pixel_centers = False
    half_pixel_centers = _execute.make_bool(half_pixel_centers, 'half_pixel_centers')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResizeBicubicGrad', grads=grads, original_image=original_image, align_corners=align_corners, half_pixel_centers=half_pixel_centers, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'align_corners', _op._get_attr_bool('align_corners'), 'half_pixel_centers', _op._get_attr_bool('half_pixel_centers'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ResizeBicubicGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result