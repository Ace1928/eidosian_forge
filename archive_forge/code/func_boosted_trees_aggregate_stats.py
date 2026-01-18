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
def boosted_trees_aggregate_stats(node_ids: _atypes.TensorFuzzingAnnotation[_atypes.Int32], gradients: _atypes.TensorFuzzingAnnotation[_atypes.Float32], hessians: _atypes.TensorFuzzingAnnotation[_atypes.Float32], feature: _atypes.TensorFuzzingAnnotation[_atypes.Int32], max_splits: int, num_buckets: int, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Aggregates the summary of accumulated stats for the batch.

  The summary stats contains gradients and hessians accumulated for each node, feature dimension id and bucket.

  Args:
    node_ids: A `Tensor` of type `int32`.
      int32; Rank 1 Tensor containing node ids for each example, shape [batch_size].
    gradients: A `Tensor` of type `float32`.
      float32; Rank 2 Tensor (shape=[batch_size, logits_dimension]) with gradients for each example.
    hessians: A `Tensor` of type `float32`.
      float32; Rank 2 Tensor (shape=[batch_size, hessian_dimension]) with hessians for each example.
    feature: A `Tensor` of type `int32`.
      int32; Rank 2 feature Tensors (shape=[batch_size, feature_dimension]).
    max_splits: An `int` that is `>= 1`.
      int; the maximum number of splits possible in the whole tree.
    num_buckets: An `int` that is `>= 1`.
      int; equals to the maximum possible value of bucketized feature.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BoostedTreesAggregateStats', name, node_ids, gradients, hessians, feature, 'max_splits', max_splits, 'num_buckets', num_buckets)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return boosted_trees_aggregate_stats_eager_fallback(node_ids, gradients, hessians, feature, max_splits=max_splits, num_buckets=num_buckets, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    max_splits = _execute.make_int(max_splits, 'max_splits')
    num_buckets = _execute.make_int(num_buckets, 'num_buckets')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BoostedTreesAggregateStats', node_ids=node_ids, gradients=gradients, hessians=hessians, feature=feature, max_splits=max_splits, num_buckets=num_buckets, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('max_splits', _op._get_attr_int('max_splits'), 'num_buckets', _op._get_attr_int('num_buckets'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BoostedTreesAggregateStats', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result