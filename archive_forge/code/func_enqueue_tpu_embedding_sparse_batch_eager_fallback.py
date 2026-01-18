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
def enqueue_tpu_embedding_sparse_batch_eager_fallback(sample_indices: List[_atypes.TensorFuzzingAnnotation[TV_EnqueueTPUEmbeddingSparseBatch_T1]], embedding_indices: List[_atypes.TensorFuzzingAnnotation[TV_EnqueueTPUEmbeddingSparseBatch_T2]], aggregation_weights: List[_atypes.TensorFuzzingAnnotation[TV_EnqueueTPUEmbeddingSparseBatch_T3]], mode_override: _atypes.TensorFuzzingAnnotation[_atypes.String], device_ordinal: int, combiners, name, ctx):
    if not isinstance(sample_indices, (list, tuple)):
        raise TypeError("Expected list for 'sample_indices' argument to 'enqueue_tpu_embedding_sparse_batch' Op, not %r." % sample_indices)
    _attr_N = len(sample_indices)
    if not isinstance(embedding_indices, (list, tuple)):
        raise TypeError("Expected list for 'embedding_indices' argument to 'enqueue_tpu_embedding_sparse_batch' Op, not %r." % embedding_indices)
    if len(embedding_indices) != _attr_N:
        raise ValueError("List argument 'embedding_indices' to 'enqueue_tpu_embedding_sparse_batch' Op with length %d must match length %d of argument 'sample_indices'." % (len(embedding_indices), _attr_N))
    if not isinstance(aggregation_weights, (list, tuple)):
        raise TypeError("Expected list for 'aggregation_weights' argument to 'enqueue_tpu_embedding_sparse_batch' Op, not %r." % aggregation_weights)
    if len(aggregation_weights) != _attr_N:
        raise ValueError("List argument 'aggregation_weights' to 'enqueue_tpu_embedding_sparse_batch' Op with length %d must match length %d of argument 'sample_indices'." % (len(aggregation_weights), _attr_N))
    if device_ordinal is None:
        device_ordinal = -1
    device_ordinal = _execute.make_int(device_ordinal, 'device_ordinal')
    if combiners is None:
        combiners = []
    if not isinstance(combiners, (list, tuple)):
        raise TypeError("Expected list for 'combiners' argument to 'enqueue_tpu_embedding_sparse_batch' Op, not %r." % combiners)
    combiners = [_execute.make_str(_s, 'combiners') for _s in combiners]
    _attr_T1, sample_indices = _execute.args_to_matching_eager(list(sample_indices), ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _attr_T2, embedding_indices = _execute.args_to_matching_eager(list(embedding_indices), ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _attr_T3, aggregation_weights = _execute.args_to_matching_eager(list(aggregation_weights), ctx, [_dtypes.float32, _dtypes.float64], _dtypes.float32)
    mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
    _inputs_flat = list(sample_indices) + list(embedding_indices) + list(aggregation_weights) + [mode_override]
    _attrs = ('T1', _attr_T1, 'T2', _attr_T2, 'T3', _attr_T3, 'N', _attr_N, 'device_ordinal', device_ordinal, 'combiners', combiners)
    _result = _execute.execute(b'EnqueueTPUEmbeddingSparseBatch', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result