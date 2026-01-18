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
def combined_non_max_suppression_eager_fallback(boxes: _atypes.TensorFuzzingAnnotation[_atypes.Float32], scores: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_output_size_per_class: _atypes.TensorFuzzingAnnotation[_atypes.Int32], max_total_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], iou_threshold: _atypes.TensorFuzzingAnnotation[_atypes.Float32], score_threshold: _atypes.TensorFuzzingAnnotation[_atypes.Float32], pad_per_class: bool, clip_boxes: bool, name, ctx):
    if pad_per_class is None:
        pad_per_class = False
    pad_per_class = _execute.make_bool(pad_per_class, 'pad_per_class')
    if clip_boxes is None:
        clip_boxes = True
    clip_boxes = _execute.make_bool(clip_boxes, 'clip_boxes')
    boxes = _ops.convert_to_tensor(boxes, _dtypes.float32)
    scores = _ops.convert_to_tensor(scores, _dtypes.float32)
    max_output_size_per_class = _ops.convert_to_tensor(max_output_size_per_class, _dtypes.int32)
    max_total_size = _ops.convert_to_tensor(max_total_size, _dtypes.int32)
    iou_threshold = _ops.convert_to_tensor(iou_threshold, _dtypes.float32)
    score_threshold = _ops.convert_to_tensor(score_threshold, _dtypes.float32)
    _inputs_flat = [boxes, scores, max_output_size_per_class, max_total_size, iou_threshold, score_threshold]
    _attrs = ('pad_per_class', pad_per_class, 'clip_boxes', clip_boxes)
    _result = _execute.execute(b'CombinedNonMaxSuppression', 4, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CombinedNonMaxSuppression', _inputs_flat, _attrs, _result)
    _result = _CombinedNonMaxSuppressionOutput._make(_result)
    return _result