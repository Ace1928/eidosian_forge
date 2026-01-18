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
def segment_max_v2(data: _atypes.TensorFuzzingAnnotation[TV_SegmentMaxV2_T], segment_ids: _atypes.TensorFuzzingAnnotation[TV_SegmentMaxV2_Tindices], num_segments: _atypes.TensorFuzzingAnnotation[TV_SegmentMaxV2_Tnumsegments], name=None) -> _atypes.TensorFuzzingAnnotation[TV_SegmentMaxV2_T]:
    """Computes the maximum along segments of a tensor.

  Read
  [the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
  for an explanation of segments.

  Computes a tensor such that
  \\\\(output_i = \\max_j(data_j)\\\\) where `max` is over `j` such
  that `segment_ids[j] == i`.

  If the maximum is empty for a given segment ID `i`, it outputs the smallest
  possible value for the specific numeric type,
  `output[i] = numeric_limits<T>::lowest()`.

  Note: That this op is currently only supported with jit_compile=True.

  Caution: On CPU, values in `segment_ids` are always validated to be sorted,
  and an error is thrown for indices that are not increasing. On GPU, this
  does not throw an error for unsorted indices. On GPU, out-of-order indices
  result in safe but unspecified behavior, which may include treating
  out-of-order indices as the same as a smaller following index.

  The only difference with SegmentMax is the additional input  `num_segments`.
  This helps in evaluating the output shape in compile time.
  `num_segments` should be consistent with segment_ids.
  e.g. Max(segment_ids) should be equal to `num_segments` - 1 for a 1-d segment_ids
  With inconsistent num_segments, the op still runs. only difference is,
  the output takes the size of num_segments irrespective of size of segment_ids and data.
  for num_segments less than expected output size, the last elements are ignored
  for num_segments more than the expected output size, last elements are assigned 
  smallest possible value for the specific numeric type.

  For example:

  >>> @tf.function(jit_compile=True)
  ... def test(c):
  ...   return tf.raw_ops.SegmentMaxV2(data=c, segment_ids=tf.constant([0, 0, 1]), num_segments=2)
  >>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
  >>> test(c).numpy()
  array([[4, 3, 3, 4],
         [5, 6, 7, 8]], dtype=int32)

  Args:
    data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
    segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.
      The values must be less than `num_segments`.

      Caution: The values are always validated to be sorted on CPU, never validated
      on GPU.
    num_segments: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SegmentMaxV2', name, data, segment_ids, num_segments)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return segment_max_v2_eager_fallback(data, segment_ids, num_segments, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SegmentMaxV2', data=data, segment_ids=segment_ids, num_segments=num_segments, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'), 'Tnumsegments', _op._get_attr_type('Tnumsegments'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SegmentMaxV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result