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
def combined_non_max_suppression(boxes: _atypes.TensorFuzzingAnnotation[_atypes.Float32], scores: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_output_size_per_class: _atypes.TensorFuzzingAnnotation[_atypes.Int32], max_total_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], iou_threshold: _atypes.TensorFuzzingAnnotation[_atypes.Float32], score_threshold: _atypes.TensorFuzzingAnnotation[_atypes.Float32], pad_per_class: bool=False, clip_boxes: bool=True, name=None):
    """Greedily selects a subset of bounding boxes in descending order of score,

  This operation performs non_max_suppression on the inputs per batch, across
  all classes.
  Prunes away boxes that have high intersection-over-union (IOU) overlap
  with previously selected boxes.  Bounding boxes are supplied as
  [y1, x1, y2, x2], where (y1, x1) and (y2, x2) are the coordinates of any
  diagonal pair of box corners and the coordinates can be provided as normalized
  (i.e., lying in the interval [0, 1]) or absolute.  Note that this algorithm
  is agnostic to where the origin is in the coordinate system. Also note that
  this algorithm is invariant to orthogonal transformations and translations
  of the coordinate system; thus translating or reflections of the coordinate
  system result in the same boxes being selected by the algorithm.
  The output of this operation is the final boxes, scores and classes tensor
  returned after performing non_max_suppression.

  Args:
    boxes: A `Tensor` of type `float32`.
      A 4-D float tensor of shape `[batch_size, num_boxes, q, 4]`. If `q` is 1 then
      same boxes are used for all classes otherwise, if `q` is equal to number of
      classes, class-specific boxes are used.
    scores: A `Tensor` of type `float32`.
      A 3-D float tensor of shape `[batch_size, num_boxes, num_classes]`
      representing a single score corresponding to each box (each row of boxes).
    max_output_size_per_class: A `Tensor` of type `int32`.
      A scalar integer tensor representing the maximum number of
      boxes to be selected by non max suppression per class
    max_total_size: A `Tensor` of type `int32`.
      An int32 scalar representing the maximum number of boxes retained over all
      classes. Note that setting this value to a large number may result in OOM error
      depending on the system workload.
    iou_threshold: A `Tensor` of type `float32`.
      A 0-D float tensor representing the threshold for deciding whether
      boxes overlap too much with respect to IOU.
    score_threshold: A `Tensor` of type `float32`.
      A 0-D float tensor representing the threshold for deciding when to remove
      boxes based on score.
    pad_per_class: An optional `bool`. Defaults to `False`.
      If false, the output nmsed boxes, scores and classes
      are padded/clipped to `max_total_size`. If true, the
      output nmsed boxes, scores and classes are padded to be of length
      `max_size_per_class`*`num_classes`, unless it exceeds `max_total_size` in
      which case it is clipped to `max_total_size`. Defaults to false.
    clip_boxes: An optional `bool`. Defaults to `True`.
      If true, assume the box coordinates are between [0, 1] and clip the output boxes
      if they fall beyond [0, 1]. If false, do not do clipping and output the box
      coordinates as it is.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections).

    nmsed_boxes: A `Tensor` of type `float32`.
    nmsed_scores: A `Tensor` of type `float32`.
    nmsed_classes: A `Tensor` of type `float32`.
    valid_detections: A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CombinedNonMaxSuppression', name, boxes, scores, max_output_size_per_class, max_total_size, iou_threshold, score_threshold, 'pad_per_class', pad_per_class, 'clip_boxes', clip_boxes)
            _result = _CombinedNonMaxSuppressionOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return combined_non_max_suppression_eager_fallback(boxes, scores, max_output_size_per_class, max_total_size, iou_threshold, score_threshold, pad_per_class=pad_per_class, clip_boxes=clip_boxes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if pad_per_class is None:
        pad_per_class = False
    pad_per_class = _execute.make_bool(pad_per_class, 'pad_per_class')
    if clip_boxes is None:
        clip_boxes = True
    clip_boxes = _execute.make_bool(clip_boxes, 'clip_boxes')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CombinedNonMaxSuppression', boxes=boxes, scores=scores, max_output_size_per_class=max_output_size_per_class, max_total_size=max_total_size, iou_threshold=iou_threshold, score_threshold=score_threshold, pad_per_class=pad_per_class, clip_boxes=clip_boxes, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('pad_per_class', _op._get_attr_bool('pad_per_class'), 'clip_boxes', _op._get_attr_bool('clip_boxes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CombinedNonMaxSuppression', _inputs_flat, _attrs, _result)
    _result = _CombinedNonMaxSuppressionOutput._make(_result)
    return _result