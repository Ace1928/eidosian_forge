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
def sparse_conditional_accumulator(dtype: TV_SparseConditionalAccumulator_dtype, shape, container: str='', shared_name: str='', reduction_type: str='MEAN', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """A conditional accumulator for aggregating sparse gradients.

  The accumulator accepts gradients marked with local_step greater or
  equal to the most recent global_step known to the accumulator. The
  average can be extracted from the accumulator, provided sufficient
  gradients have been accumulated. Extracting the average automatically
  resets the aggregate to 0, and increments the global_step recorded by
  the accumulator.

  Args:
    dtype: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.complex64, tf.int64, tf.qint8, tf.quint8, tf.qint32, tf.bfloat16, tf.qint16, tf.quint16, tf.uint16, tf.complex128, tf.half, tf.uint32, tf.uint64`.
      The type of the value being accumulated.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this accumulator is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this accumulator will be shared under the given name
      across multiple sessions.
    reduction_type: An optional `string` from: `"MEAN", "SUM"`. Defaults to `"MEAN"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("sparse_conditional_accumulator op does not support eager execution. Arg 'handle' is a ref.")
    dtype = _execute.make_type(dtype, 'dtype')
    shape = _execute.make_shape(shape, 'shape')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    if reduction_type is None:
        reduction_type = 'MEAN'
    reduction_type = _execute.make_str(reduction_type, 'reduction_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseConditionalAccumulator', dtype=dtype, shape=shape, container=container, shared_name=shared_name, reduction_type=reduction_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'shape', _op.get_attr('shape'), 'container', _op.get_attr('container'), 'shared_name', _op.get_attr('shared_name'), 'reduction_type', _op.get_attr('reduction_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseConditionalAccumulator', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result