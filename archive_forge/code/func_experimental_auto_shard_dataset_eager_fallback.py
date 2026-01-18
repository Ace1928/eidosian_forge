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
def experimental_auto_shard_dataset_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], num_workers: _atypes.TensorFuzzingAnnotation[_atypes.Int64], index: _atypes.TensorFuzzingAnnotation[_atypes.Int64], output_types, output_shapes, auto_shard_policy: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'experimental_auto_shard_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'experimental_auto_shard_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if auto_shard_policy is None:
        auto_shard_policy = 0
    auto_shard_policy = _execute.make_int(auto_shard_policy, 'auto_shard_policy')
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    num_workers = _ops.convert_to_tensor(num_workers, _dtypes.int64)
    index = _ops.convert_to_tensor(index, _dtypes.int64)
    _inputs_flat = [input_dataset, num_workers, index]
    _attrs = ('auto_shard_policy', auto_shard_policy, 'output_types', output_types, 'output_shapes', output_shapes)
    _result = _execute.execute(b'ExperimentalAutoShardDataset', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ExperimentalAutoShardDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result