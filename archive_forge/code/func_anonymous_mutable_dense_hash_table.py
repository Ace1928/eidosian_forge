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
def anonymous_mutable_dense_hash_table(empty_key: _atypes.TensorFuzzingAnnotation[TV_AnonymousMutableDenseHashTable_key_dtype], deleted_key: _atypes.TensorFuzzingAnnotation[TV_AnonymousMutableDenseHashTable_key_dtype], value_dtype: TV_AnonymousMutableDenseHashTable_value_dtype, value_shape=[], initial_num_buckets: int=131072, max_load_factor: float=0.8, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    """Creates an empty anonymous mutable hash table that uses tensors as the backing store.

  This op creates a new anonymous mutable hash table (as a resource) everytime
  it is executed, with the specified dtype of its keys and values,
  returning the resource handle. Each value must be a scalar.
  Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  It uses "open addressing" with quadratic reprobing to resolve
  collisions.

  The table is anonymous in the sense that it can only be
  accessed by the returned resource handle (e.g. it cannot be looked up
  by a name in a resource manager). The table will be automatically
  deleted when all resource handles pointing to it are gone.

  Args:
    empty_key: A `Tensor`.
      The key used to represent empty key buckets internally. Must not
      be used in insert or lookup operations.
    deleted_key: A `Tensor`. Must have the same type as `empty_key`.
    value_dtype: A `tf.DType`. Type of the table values.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The shape of each value.
    initial_num_buckets: An optional `int`. Defaults to `131072`.
      The initial number of hash table buckets. Must be a power
      to 2.
    max_load_factor: An optional `float`. Defaults to `0.8`.
      The maximum ratio between number of entries and number of
      buckets before growing the table. Must be between 0 and 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'AnonymousMutableDenseHashTable', name, empty_key, deleted_key, 'value_dtype', value_dtype, 'value_shape', value_shape, 'initial_num_buckets', initial_num_buckets, 'max_load_factor', max_load_factor)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return anonymous_mutable_dense_hash_table_eager_fallback(empty_key, deleted_key, value_dtype=value_dtype, value_shape=value_shape, initial_num_buckets=initial_num_buckets, max_load_factor=max_load_factor, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    value_dtype = _execute.make_type(value_dtype, 'value_dtype')
    if value_shape is None:
        value_shape = []
    value_shape = _execute.make_shape(value_shape, 'value_shape')
    if initial_num_buckets is None:
        initial_num_buckets = 131072
    initial_num_buckets = _execute.make_int(initial_num_buckets, 'initial_num_buckets')
    if max_load_factor is None:
        max_load_factor = 0.8
    max_load_factor = _execute.make_float(max_load_factor, 'max_load_factor')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('AnonymousMutableDenseHashTable', empty_key=empty_key, deleted_key=deleted_key, value_dtype=value_dtype, value_shape=value_shape, initial_num_buckets=initial_num_buckets, max_load_factor=max_load_factor, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('key_dtype', _op._get_attr_type('key_dtype'), 'value_dtype', _op._get_attr_type('value_dtype'), 'value_shape', _op.get_attr('value_shape'), 'initial_num_buckets', _op._get_attr_int('initial_num_buckets'), 'max_load_factor', _op.get_attr('max_load_factor'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('AnonymousMutableDenseHashTable', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result