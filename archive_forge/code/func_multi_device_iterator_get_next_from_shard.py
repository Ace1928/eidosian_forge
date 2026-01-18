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
def multi_device_iterator_get_next_from_shard(multi_device_iterator: _atypes.TensorFuzzingAnnotation[_atypes.Resource], shard_num: _atypes.TensorFuzzingAnnotation[_atypes.Int32], incarnation_id: _atypes.TensorFuzzingAnnotation[_atypes.Int64], output_types, output_shapes, name=None):
    """Gets next element for the provided shard number.

  Args:
    multi_device_iterator: A `Tensor` of type `resource`.
      A MultiDeviceIterator resource.
    shard_num: A `Tensor` of type `int32`.
      Integer representing which shard to fetch data for.
    incarnation_id: A `Tensor` of type `int64`.
      Which incarnation of the MultiDeviceIterator is running.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
      The type list for the return values.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
      The list of shapes being produced.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MultiDeviceIteratorGetNextFromShard', name, multi_device_iterator, shard_num, incarnation_id, 'output_types', output_types, 'output_shapes', output_shapes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return multi_device_iterator_get_next_from_shard_eager_fallback(multi_device_iterator, shard_num, incarnation_id, output_types=output_types, output_shapes=output_shapes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'multi_device_iterator_get_next_from_shard' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'multi_device_iterator_get_next_from_shard' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MultiDeviceIteratorGetNextFromShard', multi_device_iterator=multi_device_iterator, shard_num=shard_num, incarnation_id=incarnation_id, output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MultiDeviceIteratorGetNextFromShard', _inputs_flat, _attrs, _result)
    return _result