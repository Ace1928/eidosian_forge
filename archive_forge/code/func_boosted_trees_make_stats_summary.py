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
def boosted_trees_make_stats_summary(node_ids: _atypes.TensorFuzzingAnnotation[_atypes.Int32], gradients: _atypes.TensorFuzzingAnnotation[_atypes.Float32], hessians: _atypes.TensorFuzzingAnnotation[_atypes.Float32], bucketized_features_list: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], max_splits: int, num_buckets: int, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Makes the summary of accumulated stats for the batch.

  The summary stats contains gradients and hessians accumulated into the corresponding node and bucket for each example.

  Args:
    node_ids: A `Tensor` of type `int32`.
      int32 Rank 1 Tensor containing node ids, which each example falls into for the requested layer.
    gradients: A `Tensor` of type `float32`.
      float32; Rank 2 Tensor (shape=[#examples, 1]) for gradients.
    hessians: A `Tensor` of type `float32`.
      float32; Rank 2 Tensor (shape=[#examples, 1]) for hessians.
    bucketized_features_list: A list of at least 1 `Tensor` objects with type `int32`.
      int32 list of Rank 1 Tensors, each containing the bucketized feature (for each feature column).
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
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BoostedTreesMakeStatsSummary', name, node_ids, gradients, hessians, bucketized_features_list, 'max_splits', max_splits, 'num_buckets', num_buckets)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return boosted_trees_make_stats_summary_eager_fallback(node_ids, gradients, hessians, bucketized_features_list, max_splits=max_splits, num_buckets=num_buckets, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(bucketized_features_list, (list, tuple)):
        raise TypeError("Expected list for 'bucketized_features_list' argument to 'boosted_trees_make_stats_summary' Op, not %r." % bucketized_features_list)
    _attr_num_features = len(bucketized_features_list)
    max_splits = _execute.make_int(max_splits, 'max_splits')
    num_buckets = _execute.make_int(num_buckets, 'num_buckets')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BoostedTreesMakeStatsSummary', node_ids=node_ids, gradients=gradients, hessians=hessians, bucketized_features_list=bucketized_features_list, max_splits=max_splits, num_buckets=num_buckets, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('max_splits', _op._get_attr_int('max_splits'), 'num_buckets', _op._get_attr_int('num_buckets'), 'num_features', _op._get_attr_int('num_features'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BoostedTreesMakeStatsSummary', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result