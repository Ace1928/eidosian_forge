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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('image.rgb_to_hsv')
def rgb_to_hsv(images: _atypes.TensorFuzzingAnnotation[TV_RGBToHSV_T], name=None) -> _atypes.TensorFuzzingAnnotation[TV_RGBToHSV_T]:
    """Converts one or more images from RGB to HSV.

  Outputs a tensor of the same shape as the `images` tensor, containing the HSV
  value of the pixels. The output is only well defined if the value in `images`
  are in `[0,1]`.

  `output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
  `output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
  corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.

  Usage Example:

  >>> blue_image = tf.stack([
  ...    tf.zeros([5,5]),
  ...    tf.zeros([5,5]),
  ...    tf.ones([5,5])],
  ...    axis=-1)
  >>> blue_hsv_image = tf.image.rgb_to_hsv(blue_image)
  >>> blue_hsv_image[0,0].numpy()
  array([0.6666667, 1. , 1. ], dtype=float32)

  Args:
    images: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      1-D or higher rank. RGB data to convert. Last dimension must be size 3.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RGBToHSV', name, images)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_rgb_to_hsv((images, name), None)
            if _result is not NotImplemented:
                return _result
            return rgb_to_hsv_eager_fallback(images, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(rgb_to_hsv, (), dict(images=images, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_rgb_to_hsv((images, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('RGBToHSV', images=images, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(rgb_to_hsv, (), dict(images=images, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RGBToHSV', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result