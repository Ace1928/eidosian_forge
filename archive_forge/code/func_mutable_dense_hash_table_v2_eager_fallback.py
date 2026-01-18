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
def mutable_dense_hash_table_v2_eager_fallback(empty_key: _atypes.TensorFuzzingAnnotation[TV_MutableDenseHashTableV2_key_dtype], deleted_key: _atypes.TensorFuzzingAnnotation[TV_MutableDenseHashTableV2_key_dtype], value_dtype: TV_MutableDenseHashTableV2_value_dtype, container: str, shared_name: str, use_node_name_sharing: bool, value_shape, initial_num_buckets: int, max_load_factor: float, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
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
    if value_shape is None:
        value_shape = []
    value_shape = _execute.make_shape(value_shape, 'value_shape')
    if initial_num_buckets is None:
        initial_num_buckets = 131072
    initial_num_buckets = _execute.make_int(initial_num_buckets, 'initial_num_buckets')
    if max_load_factor is None:
        max_load_factor = 0.8
    max_load_factor = _execute.make_float(max_load_factor, 'max_load_factor')
    _attr_key_dtype, _inputs_key_dtype = _execute.args_to_matching_eager([empty_key, deleted_key], ctx, [])
    empty_key, deleted_key = _inputs_key_dtype
    _inputs_flat = [empty_key, deleted_key]
    _attrs = ('container', container, 'shared_name', shared_name, 'use_node_name_sharing', use_node_name_sharing, 'key_dtype', _attr_key_dtype, 'value_dtype', value_dtype, 'value_shape', value_shape, 'initial_num_buckets', initial_num_buckets, 'max_load_factor', max_load_factor)
    _result = _execute.execute(b'MutableDenseHashTableV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('MutableDenseHashTableV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result