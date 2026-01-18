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
def anonymous_mutable_hash_table_of_tensors(key_dtype: TV_AnonymousMutableHashTableOfTensors_key_dtype, value_dtype: TV_AnonymousMutableHashTableOfTensors_value_dtype, value_shape=[], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    """Creates an empty anonymous mutable hash table of vector values.

  This op creates a new anonymous mutable hash table (as a resource) everytime
  it is executed, with the specified dtype of its keys and values,
  returning the resource handle. Each value must be a vector.
  Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.
  The table is anonymous in the sense that it can only be
  accessed by the returned resource handle (e.g. it cannot be looked up
  by a name in a resource manager). The table will be automatically
  deleted when all resource handles pointing to it are gone.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    value_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'AnonymousMutableHashTableOfTensors', name, 'key_dtype', key_dtype, 'value_dtype', value_dtype, 'value_shape', value_shape)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return anonymous_mutable_hash_table_of_tensors_eager_fallback(key_dtype=key_dtype, value_dtype=value_dtype, value_shape=value_shape, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    key_dtype = _execute.make_type(key_dtype, 'key_dtype')
    value_dtype = _execute.make_type(value_dtype, 'value_dtype')
    if value_shape is None:
        value_shape = []
    value_shape = _execute.make_shape(value_shape, 'value_shape')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('AnonymousMutableHashTableOfTensors', key_dtype=key_dtype, value_dtype=value_dtype, value_shape=value_shape, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('key_dtype', _op._get_attr_type('key_dtype'), 'value_dtype', _op._get_attr_type('value_dtype'), 'value_shape', _op.get_attr('value_shape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('AnonymousMutableHashTableOfTensors', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result