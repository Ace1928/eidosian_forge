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
def sparse_segment_sqrt_n_with_num_segments(data: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSqrtNWithNumSegments_T], indices: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSqrtNWithNumSegments_Tidx], segment_ids: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSqrtNWithNumSegments_Tsegmentids], num_segments: _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSqrtNWithNumSegments_Tnumsegments], name=None) -> _atypes.TensorFuzzingAnnotation[TV_SparseSegmentSqrtNWithNumSegments_T]:
    """Computes the sum along sparse segments of a tensor divided by the sqrt of N.

  N is the size of the segment being reduced.

  Like `SparseSegmentSqrtN`, but allows missing ids in `segment_ids`. If an id is
  missing, the `output` tensor at that position will be zeroed.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Args:
    data: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Has same rank as `segment_ids`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor. Values should be sorted and can be repeated.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Should equal the number of distinct segment IDs.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseSegmentSqrtNWithNumSegments', name, data, indices, segment_ids, num_segments)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_segment_sqrt_n_with_num_segments_eager_fallback(data, indices, segment_ids, num_segments, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseSegmentSqrtNWithNumSegments', data=data, indices=indices, segment_ids=segment_ids, num_segments=num_segments, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tidx', _op._get_attr_type('Tidx'), 'Tnumsegments', _op._get_attr_type('Tnumsegments'), 'Tsegmentids', _op._get_attr_type('Tsegmentids'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseSegmentSqrtNWithNumSegments', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result