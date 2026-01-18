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
def sparse_accumulator_take_gradient(handle: _atypes.TensorFuzzingAnnotation[_atypes.String], num_required: _atypes.TensorFuzzingAnnotation[_atypes.Int32], dtype: TV_SparseAccumulatorTakeGradient_dtype, name=None):
    """Extracts the average sparse gradient in a SparseConditionalAccumulator.

  The op will blocks until sufficient (i.e., more than num_required)
  gradients have been accumulated. If the accumulator has already
  aggregated more than num_required gradients, it will return its
  average of the accumulated gradients.  Also automatically increments
  the recorded global_step in the accumulator by 1, and resets the
  aggregate to 0.

  Args:
    handle: A `Tensor` of type mutable `string`.
      The handle to a SparseConditionalAccumulator.
    num_required: A `Tensor` of type `int32`.
      Number of gradients required before we return an aggregate.
    dtype: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.complex64, tf.int64, tf.qint8, tf.quint8, tf.qint32, tf.bfloat16, tf.qint16, tf.quint16, tf.uint16, tf.complex128, tf.half, tf.uint32, tf.uint64`.
      The data type of accumulated gradients. Needs to correspond to the type
      of the accumulator.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (indices, values, shape).

    indices: A `Tensor` of type `int64`.
    values: A `Tensor` of type `dtype`.
    shape: A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("sparse_accumulator_take_gradient op does not support eager execution. Arg 'handle' is a ref.")
    dtype = _execute.make_type(dtype, 'dtype')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseAccumulatorTakeGradient', handle=handle, num_required=num_required, dtype=dtype, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseAccumulatorTakeGradient', _inputs_flat, _attrs, _result)
    _result = _SparseAccumulatorTakeGradientOutput._make(_result)
    return _result