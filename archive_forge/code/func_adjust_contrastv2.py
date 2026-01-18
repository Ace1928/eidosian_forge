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
def adjust_contrastv2(images: _atypes.TensorFuzzingAnnotation[TV_AdjustContrastv2_T], contrast_factor: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name=None) -> _atypes.TensorFuzzingAnnotation[TV_AdjustContrastv2_T]:
    """Adjust the contrast of one or more images.

  `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
  interpreted as `[height, width, channels]`.  The other dimensions only
  represent a collection of images, such as `[batch, height, width, channels].`

  Contrast is adjusted independently for each channel of each image.

  For each channel, the Op first computes the mean of the image pixels in the
  channel and then adjusts each component of each pixel to
  `(x - mean) * contrast_factor + mean`.

  Args:
    images: A `Tensor`. Must be one of the following types: `half`, `float32`.
      Images to adjust.  At least 3-D.
    contrast_factor: A `Tensor` of type `float32`.
      A float multiplier for adjusting contrast.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'AdjustContrastv2', name, images, contrast_factor)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return adjust_contrastv2_eager_fallback(images, contrast_factor, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('AdjustContrastv2', images=images, contrast_factor=contrast_factor, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('AdjustContrastv2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result