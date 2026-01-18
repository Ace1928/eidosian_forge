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
def draw_bounding_boxes(images: _atypes.TensorFuzzingAnnotation[TV_DrawBoundingBoxes_T], boxes: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name=None) -> _atypes.TensorFuzzingAnnotation[TV_DrawBoundingBoxes_T]:
    """Draw bounding boxes on a batch of images.

  Outputs a copy of `images` but draws on top of the pixels zero or more bounding
  boxes specified by the locations in `boxes`. The coordinates of the each
  bounding box in `boxes` are encoded as `[y_min, x_min, y_max, x_max]`. The
  bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
  height of the underlying image.

  For example, if an image is 100 x 200 pixels (height x width) and the bounding
  box is `[0.1, 0.2, 0.5, 0.9]`, the upper-left and bottom-right coordinates of
  the bounding box will be `(40, 10)` to `(180, 50)` (in (x,y) coordinates).

  Parts of the bounding box may fall outside the image.

  Args:
    images: A `Tensor`. Must be one of the following types: `float32`, `half`.
      4-D with shape `[batch, height, width, depth]`. A batch of images.
    boxes: A `Tensor` of type `float32`.
      3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding
      boxes.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DrawBoundingBoxes', name, images, boxes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return draw_bounding_boxes_eager_fallback(images, boxes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DrawBoundingBoxes', images=images, boxes=boxes, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DrawBoundingBoxes', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result