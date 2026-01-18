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
def crop_and_resize_grad_boxes(grads: _atypes.TensorFuzzingAnnotation[_atypes.Float32], image: _atypes.TensorFuzzingAnnotation[TV_CropAndResizeGradBoxes_T], boxes: _atypes.TensorFuzzingAnnotation[_atypes.Float32], box_ind: _atypes.TensorFuzzingAnnotation[_atypes.Int32], method: str='bilinear', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Computes the gradient of the crop_and_resize op wrt the input boxes tensor.

  Args:
    grads: A `Tensor` of type `float32`.
      A 4-D tensor of shape `[num_boxes, crop_height, crop_width, depth]`.
    image: A `Tensor`. Must be one of the following types: `uint8`, `uint16`, `int8`, `int16`, `int32`, `int64`, `half`, `float32`, `float64`.
      A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32`.
      A 2-D tensor of shape `[num_boxes, 4]`. The `i`-th row of the tensor
      specifies the coordinates of a box in the `box_ind[i]` image and is specified
      in normalized coordinates `[y1, x1, y2, x2]`. A normalized coordinate value of
      `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
      `[0, 1]` interval of normalized image height is mapped to
      `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
      which case the sampled crop is an up-down flipped version of the original
      image. The width dimension is treated similarly. Normalized coordinates
      outside the `[0, 1]` range are allowed, in which case we use
      `extrapolation_value` to extrapolate the input image values.
    box_ind: A `Tensor` of type `int32`.
      A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
      The value of `box_ind[i]` specifies the image that the `i`-th box refers to.
    method: An optional `string` from: `"bilinear"`. Defaults to `"bilinear"`.
      A string specifying the interpolation method. Only 'bilinear' is
      supported for now.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CropAndResizeGradBoxes', name, grads, image, boxes, box_ind, 'method', method)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return crop_and_resize_grad_boxes_eager_fallback(grads, image, boxes, box_ind, method=method, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if method is None:
        method = 'bilinear'
    method = _execute.make_str(method, 'method')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CropAndResizeGradBoxes', grads=grads, image=image, boxes=boxes, box_ind=box_ind, method=method, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'method', _op.get_attr('method'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CropAndResizeGradBoxes', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result