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
def boosted_trees_aggregate_stats_eager_fallback(node_ids: _atypes.TensorFuzzingAnnotation[_atypes.Int32], gradients: _atypes.TensorFuzzingAnnotation[_atypes.Float32], hessians: _atypes.TensorFuzzingAnnotation[_atypes.Float32], feature: _atypes.TensorFuzzingAnnotation[_atypes.Int32], max_splits: int, num_buckets: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    max_splits = _execute.make_int(max_splits, 'max_splits')
    num_buckets = _execute.make_int(num_buckets, 'num_buckets')
    node_ids = _ops.convert_to_tensor(node_ids, _dtypes.int32)
    gradients = _ops.convert_to_tensor(gradients, _dtypes.float32)
    hessians = _ops.convert_to_tensor(hessians, _dtypes.float32)
    feature = _ops.convert_to_tensor(feature, _dtypes.int32)
    _inputs_flat = [node_ids, gradients, hessians, feature]
    _attrs = ('max_splits', max_splits, 'num_buckets', num_buckets)
    _result = _execute.execute(b'BoostedTreesAggregateStats', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BoostedTreesAggregateStats', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result