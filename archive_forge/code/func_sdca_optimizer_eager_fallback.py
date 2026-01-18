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
def sdca_optimizer_eager_fallback(sparse_example_indices: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], sparse_feature_indices: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], sparse_feature_values: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], dense_features: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], example_weights: _atypes.TensorFuzzingAnnotation[_atypes.Float32], example_labels: _atypes.TensorFuzzingAnnotation[_atypes.Float32], sparse_indices: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], sparse_weights: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], dense_weights: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], example_state_data: _atypes.TensorFuzzingAnnotation[_atypes.Float32], loss_type: str, l1: float, l2: float, num_loss_partitions: int, num_inner_iterations: int, adaptative: bool, name, ctx):
    if not isinstance(sparse_example_indices, (list, tuple)):
        raise TypeError("Expected list for 'sparse_example_indices' argument to 'sdca_optimizer' Op, not %r." % sparse_example_indices)
    _attr_num_sparse_features = len(sparse_example_indices)
    if not isinstance(sparse_feature_indices, (list, tuple)):
        raise TypeError("Expected list for 'sparse_feature_indices' argument to 'sdca_optimizer' Op, not %r." % sparse_feature_indices)
    if len(sparse_feature_indices) != _attr_num_sparse_features:
        raise ValueError("List argument 'sparse_feature_indices' to 'sdca_optimizer' Op with length %d must match length %d of argument 'sparse_example_indices'." % (len(sparse_feature_indices), _attr_num_sparse_features))
    if not isinstance(sparse_indices, (list, tuple)):
        raise TypeError("Expected list for 'sparse_indices' argument to 'sdca_optimizer' Op, not %r." % sparse_indices)
    if len(sparse_indices) != _attr_num_sparse_features:
        raise ValueError("List argument 'sparse_indices' to 'sdca_optimizer' Op with length %d must match length %d of argument 'sparse_example_indices'." % (len(sparse_indices), _attr_num_sparse_features))
    if not isinstance(sparse_weights, (list, tuple)):
        raise TypeError("Expected list for 'sparse_weights' argument to 'sdca_optimizer' Op, not %r." % sparse_weights)
    if len(sparse_weights) != _attr_num_sparse_features:
        raise ValueError("List argument 'sparse_weights' to 'sdca_optimizer' Op with length %d must match length %d of argument 'sparse_example_indices'." % (len(sparse_weights), _attr_num_sparse_features))
    if not isinstance(sparse_feature_values, (list, tuple)):
        raise TypeError("Expected list for 'sparse_feature_values' argument to 'sdca_optimizer' Op, not %r." % sparse_feature_values)
    _attr_num_sparse_features_with_values = len(sparse_feature_values)
    if not isinstance(dense_features, (list, tuple)):
        raise TypeError("Expected list for 'dense_features' argument to 'sdca_optimizer' Op, not %r." % dense_features)
    _attr_num_dense_features = len(dense_features)
    if not isinstance(dense_weights, (list, tuple)):
        raise TypeError("Expected list for 'dense_weights' argument to 'sdca_optimizer' Op, not %r." % dense_weights)
    if len(dense_weights) != _attr_num_dense_features:
        raise ValueError("List argument 'dense_weights' to 'sdca_optimizer' Op with length %d must match length %d of argument 'dense_features'." % (len(dense_weights), _attr_num_dense_features))
    loss_type = _execute.make_str(loss_type, 'loss_type')
    l1 = _execute.make_float(l1, 'l1')
    l2 = _execute.make_float(l2, 'l2')
    num_loss_partitions = _execute.make_int(num_loss_partitions, 'num_loss_partitions')
    num_inner_iterations = _execute.make_int(num_inner_iterations, 'num_inner_iterations')
    if adaptative is None:
        adaptative = True
    adaptative = _execute.make_bool(adaptative, 'adaptative')
    sparse_example_indices = _ops.convert_n_to_tensor(sparse_example_indices, _dtypes.int64)
    sparse_feature_indices = _ops.convert_n_to_tensor(sparse_feature_indices, _dtypes.int64)
    sparse_feature_values = _ops.convert_n_to_tensor(sparse_feature_values, _dtypes.float32)
    dense_features = _ops.convert_n_to_tensor(dense_features, _dtypes.float32)
    example_weights = _ops.convert_to_tensor(example_weights, _dtypes.float32)
    example_labels = _ops.convert_to_tensor(example_labels, _dtypes.float32)
    sparse_indices = _ops.convert_n_to_tensor(sparse_indices, _dtypes.int64)
    sparse_weights = _ops.convert_n_to_tensor(sparse_weights, _dtypes.float32)
    dense_weights = _ops.convert_n_to_tensor(dense_weights, _dtypes.float32)
    example_state_data = _ops.convert_to_tensor(example_state_data, _dtypes.float32)
    _inputs_flat = list(sparse_example_indices) + list(sparse_feature_indices) + list(sparse_feature_values) + list(dense_features) + [example_weights, example_labels] + list(sparse_indices) + list(sparse_weights) + list(dense_weights) + [example_state_data]
    _attrs = ('loss_type', loss_type, 'adaptative', adaptative, 'num_sparse_features', _attr_num_sparse_features, 'num_sparse_features_with_values', _attr_num_sparse_features_with_values, 'num_dense_features', _attr_num_dense_features, 'l1', l1, 'l2', l2, 'num_loss_partitions', num_loss_partitions, 'num_inner_iterations', num_inner_iterations)
    _result = _execute.execute(b'SdcaOptimizer', _attr_num_sparse_features + _attr_num_dense_features + 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SdcaOptimizer', _inputs_flat, _attrs, _result)
    _result = _result[:1] + [_result[1:1 + _attr_num_sparse_features]] + _result[1 + _attr_num_sparse_features:]
    _result = _result[:2] + [_result[2:]]
    _result = _SdcaOptimizerOutput._make(_result)
    return _result