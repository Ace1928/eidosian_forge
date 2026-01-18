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
def strided_slice_grad_eager_fallback(shape: _atypes.TensorFuzzingAnnotation[TV_StridedSliceGrad_Index], begin: _atypes.TensorFuzzingAnnotation[TV_StridedSliceGrad_Index], end: _atypes.TensorFuzzingAnnotation[TV_StridedSliceGrad_Index], strides: _atypes.TensorFuzzingAnnotation[TV_StridedSliceGrad_Index], dy: _atypes.TensorFuzzingAnnotation[TV_StridedSliceGrad_T], begin_mask: int, end_mask: int, ellipsis_mask: int, new_axis_mask: int, shrink_axis_mask: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_StridedSliceGrad_T]:
    if begin_mask is None:
        begin_mask = 0
    begin_mask = _execute.make_int(begin_mask, 'begin_mask')
    if end_mask is None:
        end_mask = 0
    end_mask = _execute.make_int(end_mask, 'end_mask')
    if ellipsis_mask is None:
        ellipsis_mask = 0
    ellipsis_mask = _execute.make_int(ellipsis_mask, 'ellipsis_mask')
    if new_axis_mask is None:
        new_axis_mask = 0
    new_axis_mask = _execute.make_int(new_axis_mask, 'new_axis_mask')
    if shrink_axis_mask is None:
        shrink_axis_mask = 0
    shrink_axis_mask = _execute.make_int(shrink_axis_mask, 'shrink_axis_mask')
    _attr_T, (dy,) = _execute.args_to_matching_eager([dy], ctx, [])
    _attr_Index, _inputs_Index = _execute.args_to_matching_eager([shape, begin, end, strides], ctx, [_dtypes.int32, _dtypes.int64])
    shape, begin, end, strides = _inputs_Index
    _inputs_flat = [shape, begin, end, strides, dy]
    _attrs = ('T', _attr_T, 'Index', _attr_Index, 'begin_mask', begin_mask, 'end_mask', end_mask, 'ellipsis_mask', ellipsis_mask, 'new_axis_mask', new_axis_mask, 'shrink_axis_mask', shrink_axis_mask)
    _result = _execute.execute(b'StridedSliceGrad', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('StridedSliceGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result