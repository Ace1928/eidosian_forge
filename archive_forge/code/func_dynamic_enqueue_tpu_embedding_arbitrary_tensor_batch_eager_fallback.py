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
def dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch_eager_fallback(sample_indices_or_row_splits: List[_atypes.TensorFuzzingAnnotation[TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T1]], embedding_indices: List[_atypes.TensorFuzzingAnnotation[TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T2]], aggregation_weights: List[_atypes.TensorFuzzingAnnotation[TV_DynamicEnqueueTPUEmbeddingArbitraryTensorBatch_T3]], mode_override: _atypes.TensorFuzzingAnnotation[_atypes.String], device_ordinal: _atypes.TensorFuzzingAnnotation[_atypes.Int32], combiners, name, ctx):
    if not isinstance(sample_indices_or_row_splits, (list, tuple)):
        raise TypeError("Expected list for 'sample_indices_or_row_splits' argument to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % sample_indices_or_row_splits)
    _attr_N = len(sample_indices_or_row_splits)
    if not isinstance(embedding_indices, (list, tuple)):
        raise TypeError("Expected list for 'embedding_indices' argument to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % embedding_indices)
    if len(embedding_indices) != _attr_N:
        raise ValueError("List argument 'embedding_indices' to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d must match length %d of argument 'sample_indices_or_row_splits'." % (len(embedding_indices), _attr_N))
    if not isinstance(aggregation_weights, (list, tuple)):
        raise TypeError("Expected list for 'aggregation_weights' argument to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % aggregation_weights)
    if len(aggregation_weights) != _attr_N:
        raise ValueError("List argument 'aggregation_weights' to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op with length %d must match length %d of argument 'sample_indices_or_row_splits'." % (len(aggregation_weights), _attr_N))
    if combiners is None:
        combiners = []
    if not isinstance(combiners, (list, tuple)):
        raise TypeError("Expected list for 'combiners' argument to 'dynamic_enqueue_tpu_embedding_arbitrary_tensor_batch' Op, not %r." % combiners)
    combiners = [_execute.make_str(_s, 'combiners') for _s in combiners]
    _attr_T1, sample_indices_or_row_splits = _execute.args_to_matching_eager(list(sample_indices_or_row_splits), ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _attr_T2, embedding_indices = _execute.args_to_matching_eager(list(embedding_indices), ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _attr_T3, aggregation_weights = _execute.args_to_matching_eager(list(aggregation_weights), ctx, [_dtypes.float32, _dtypes.float64], _dtypes.float32)
    mode_override = _ops.convert_to_tensor(mode_override, _dtypes.string)
    device_ordinal = _ops.convert_to_tensor(device_ordinal, _dtypes.int32)
    _inputs_flat = list(sample_indices_or_row_splits) + list(embedding_indices) + list(aggregation_weights) + [mode_override, device_ordinal]
    _attrs = ('T1', _attr_T1, 'T2', _attr_T2, 'T3', _attr_T3, 'N', _attr_N, 'combiners', combiners)
    _result = _execute.execute(b'DynamicEnqueueTPUEmbeddingArbitraryTensorBatch', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result