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
def non_max_suppression_v4_eager_fallback(boxes: _atypes.TensorFuzzingAnnotation[TV_NonMaxSuppressionV4_T], scores: _atypes.TensorFuzzingAnnotation[TV_NonMaxSuppressionV4_T], max_output_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], iou_threshold: _atypes.TensorFuzzingAnnotation[TV_NonMaxSuppressionV4_T_threshold], score_threshold: _atypes.TensorFuzzingAnnotation[TV_NonMaxSuppressionV4_T_threshold], pad_to_max_output_size: bool, name, ctx):
    if pad_to_max_output_size is None:
        pad_to_max_output_size = False
    pad_to_max_output_size = _execute.make_bool(pad_to_max_output_size, 'pad_to_max_output_size')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([boxes, scores], ctx, [_dtypes.half, _dtypes.float32], _dtypes.float32)
    boxes, scores = _inputs_T
    _attr_T_threshold, _inputs_T_threshold = _execute.args_to_matching_eager([iou_threshold, score_threshold], ctx, [_dtypes.half, _dtypes.float32], _dtypes.float32)
    iou_threshold, score_threshold = _inputs_T_threshold
    max_output_size = _ops.convert_to_tensor(max_output_size, _dtypes.int32)
    _inputs_flat = [boxes, scores, max_output_size, iou_threshold, score_threshold]
    _attrs = ('T', _attr_T, 'T_threshold', _attr_T_threshold, 'pad_to_max_output_size', pad_to_max_output_size)
    _result = _execute.execute(b'NonMaxSuppressionV4', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('NonMaxSuppressionV4', _inputs_flat, _attrs, _result)
    _result = _NonMaxSuppressionV4Output._make(_result)
    return _result