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
def mutable_hash_table_v2(key_dtype: TV_MutableHashTableV2_key_dtype, value_dtype: TV_MutableHashTableV2_value_dtype, container: str='', shared_name: str='', use_node_name_sharing: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    """Creates an empty hash table.

  This op creates a mutable hash table, specifying the type of its keys and
  values. Each value must be a scalar. Data can be inserted into the table using
  the insert operations. It does not support the initialization operation.

  Args:
    key_dtype: A `tf.DType`. Type of the table keys.
    value_dtype: A `tf.DType`. Type of the table values.
    container: An optional `string`. Defaults to `""`.
      If non-empty, this table is placed in the given container.
      Otherwise, a default container is used.
    shared_name: An optional `string`. Defaults to `""`.
      If non-empty, this table is shared under the given name across
      multiple sessions.
    use_node_name_sharing: An optional `bool`. Defaults to `False`.
      If true and shared_name is empty, the table is shared
      using the node name.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MutableHashTableV2', name, 'container', container, 'shared_name', shared_name, 'use_node_name_sharing', use_node_name_sharing, 'key_dtype', key_dtype, 'value_dtype', value_dtype)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return mutable_hash_table_v2_eager_fallback(container=container, shared_name=shared_name, use_node_name_sharing=use_node_name_sharing, key_dtype=key_dtype, value_dtype=value_dtype, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    key_dtype = _execute.make_type(key_dtype, 'key_dtype')
    value_dtype = _execute.make_type(value_dtype, 'value_dtype')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    if use_node_name_sharing is None:
        use_node_name_sharing = False
    use_node_name_sharing = _execute.make_bool(use_node_name_sharing, 'use_node_name_sharing')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MutableHashTableV2', key_dtype=key_dtype, value_dtype=value_dtype, container=container, shared_name=shared_name, use_node_name_sharing=use_node_name_sharing, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('container', _op.get_attr('container'), 'shared_name', _op.get_attr('shared_name'), 'use_node_name_sharing', _op._get_attr_bool('use_node_name_sharing'), 'key_dtype', _op._get_attr_type('key_dtype'), 'value_dtype', _op._get_attr_type('value_dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MutableHashTableV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result