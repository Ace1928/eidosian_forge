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
def mutable_hash_table_v2_eager_fallback(key_dtype: TV_MutableHashTableV2_key_dtype, value_dtype: TV_MutableHashTableV2_value_dtype, container: str, shared_name: str, use_node_name_sharing: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
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
    _inputs_flat = []
    _attrs = ('container', container, 'shared_name', shared_name, 'use_node_name_sharing', use_node_name_sharing, 'key_dtype', key_dtype, 'value_dtype', value_dtype)
    _result = _execute.execute(b'MutableHashTableV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('MutableHashTableV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result