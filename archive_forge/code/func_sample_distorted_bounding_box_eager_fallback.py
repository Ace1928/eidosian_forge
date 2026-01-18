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
def sample_distorted_bounding_box_eager_fallback(image_size: _atypes.TensorFuzzingAnnotation[TV_SampleDistortedBoundingBox_T], bounding_boxes: _atypes.TensorFuzzingAnnotation[_atypes.Float32], seed: int, seed2: int, min_object_covered: float, aspect_ratio_range, area_range, max_attempts: int, use_image_if_no_bounding_boxes: bool, name, ctx):
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    if min_object_covered is None:
        min_object_covered = 0.1
    min_object_covered = _execute.make_float(min_object_covered, 'min_object_covered')
    if aspect_ratio_range is None:
        aspect_ratio_range = [0.75, 1.33]
    if not isinstance(aspect_ratio_range, (list, tuple)):
        raise TypeError("Expected list for 'aspect_ratio_range' argument to 'sample_distorted_bounding_box' Op, not %r." % aspect_ratio_range)
    aspect_ratio_range = [_execute.make_float(_f, 'aspect_ratio_range') for _f in aspect_ratio_range]
    if area_range is None:
        area_range = [0.05, 1]
    if not isinstance(area_range, (list, tuple)):
        raise TypeError("Expected list for 'area_range' argument to 'sample_distorted_bounding_box' Op, not %r." % area_range)
    area_range = [_execute.make_float(_f, 'area_range') for _f in area_range]
    if max_attempts is None:
        max_attempts = 100
    max_attempts = _execute.make_int(max_attempts, 'max_attempts')
    if use_image_if_no_bounding_boxes is None:
        use_image_if_no_bounding_boxes = False
    use_image_if_no_bounding_boxes = _execute.make_bool(use_image_if_no_bounding_boxes, 'use_image_if_no_bounding_boxes')
    _attr_T, (image_size,) = _execute.args_to_matching_eager([image_size], ctx, [_dtypes.uint8, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64])
    bounding_boxes = _ops.convert_to_tensor(bounding_boxes, _dtypes.float32)
    _inputs_flat = [image_size, bounding_boxes]
    _attrs = ('T', _attr_T, 'seed', seed, 'seed2', seed2, 'min_object_covered', min_object_covered, 'aspect_ratio_range', aspect_ratio_range, 'area_range', area_range, 'max_attempts', max_attempts, 'use_image_if_no_bounding_boxes', use_image_if_no_bounding_boxes)
    _result = _execute.execute(b'SampleDistortedBoundingBox', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SampleDistortedBoundingBox', _inputs_flat, _attrs, _result)
    _result = _SampleDistortedBoundingBoxOutput._make(_result)
    return _result