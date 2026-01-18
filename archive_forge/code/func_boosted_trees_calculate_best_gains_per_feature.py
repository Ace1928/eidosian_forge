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
def boosted_trees_calculate_best_gains_per_feature(node_id_range: _atypes.TensorFuzzingAnnotation[_atypes.Int32], stats_summary_list: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], l1: _atypes.TensorFuzzingAnnotation[_atypes.Float32], l2: _atypes.TensorFuzzingAnnotation[_atypes.Float32], tree_complexity: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_node_weight: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_splits: int, name=None):
    """Calculates gains for each feature and returns the best possible split information for the feature.

  The split information is the best threshold (bucket id), gains and left/right node contributions per node for each feature.

  It is possible that not all nodes can be split on each feature. Hence, the list of possible nodes can differ between the features. Therefore, we return `node_ids_list` for each feature, containing the list of nodes that this feature can be used to split.

  In this manner, the output is the best split per features and per node, so that it needs to be combined later to produce the best split for each node (among all possible features).

  The length of output lists are all of the same length, `num_features`.
  The output shapes are compatible in a way that the first dimension of all tensors of all lists are the same and equal to the number of possible split nodes for each feature.

  Args:
    node_id_range: A `Tensor` of type `int32`.
      A Rank 1 tensor (shape=[2]) to specify the range [first, last) of node ids to process within `stats_summary_list`. The nodes are iterated between the two nodes specified by the tensor, as like `for node_id in range(node_id_range[0], node_id_range[1])` (Note that the last index node_id_range[1] is exclusive).
    stats_summary_list: A list of at least 1 `Tensor` objects with type `float32`.
      A list of Rank 3 tensor (#shape=[max_splits, bucket, 2]) for accumulated stats summary (gradient/hessian) per node per buckets for each feature. The first dimension of the tensor is the maximum number of splits, and thus not all elements of it will be used, but only the indexes specified by node_ids will be used.
    l1: A `Tensor` of type `float32`.
      l1 regularization factor on leaf weights, per instance based.
    l2: A `Tensor` of type `float32`.
      l2 regularization factor on leaf weights, per instance based.
    tree_complexity: A `Tensor` of type `float32`.
      adjustment to the gain, per leaf based.
    min_node_weight: A `Tensor` of type `float32`.
      minimum avg of hessians in a node before required for the node to be considered for splitting.
    max_splits: An `int` that is `>= 1`.
      the number of nodes that can be split in the whole tree. Used as a dimension of output tensors.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (node_ids_list, gains_list, thresholds_list, left_node_contribs_list, right_node_contribs_list).

    node_ids_list: A list with the same length as `stats_summary_list` of `Tensor` objects with type `int32`.
    gains_list: A list with the same length as `stats_summary_list` of `Tensor` objects with type `float32`.
    thresholds_list: A list with the same length as `stats_summary_list` of `Tensor` objects with type `int32`.
    left_node_contribs_list: A list with the same length as `stats_summary_list` of `Tensor` objects with type `float32`.
    right_node_contribs_list: A list with the same length as `stats_summary_list` of `Tensor` objects with type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BoostedTreesCalculateBestGainsPerFeature', name, node_id_range, stats_summary_list, l1, l2, tree_complexity, min_node_weight, 'max_splits', max_splits)
            _result = _BoostedTreesCalculateBestGainsPerFeatureOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return boosted_trees_calculate_best_gains_per_feature_eager_fallback(node_id_range, stats_summary_list, l1, l2, tree_complexity, min_node_weight, max_splits=max_splits, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(stats_summary_list, (list, tuple)):
        raise TypeError("Expected list for 'stats_summary_list' argument to 'boosted_trees_calculate_best_gains_per_feature' Op, not %r." % stats_summary_list)
    _attr_num_features = len(stats_summary_list)
    max_splits = _execute.make_int(max_splits, 'max_splits')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BoostedTreesCalculateBestGainsPerFeature', node_id_range=node_id_range, stats_summary_list=stats_summary_list, l1=l1, l2=l2, tree_complexity=tree_complexity, min_node_weight=min_node_weight, max_splits=max_splits, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('max_splits', _op._get_attr_int('max_splits'), 'num_features', _op._get_attr_int('num_features'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BoostedTreesCalculateBestGainsPerFeature', _inputs_flat, _attrs, _result)
    _result = [_result[:_attr_num_features]] + _result[_attr_num_features:]
    _result = _result[:1] + [_result[1:1 + _attr_num_features]] + _result[1 + _attr_num_features:]
    _result = _result[:2] + [_result[2:2 + _attr_num_features]] + _result[2 + _attr_num_features:]
    _result = _result[:3] + [_result[3:3 + _attr_num_features]] + _result[3 + _attr_num_features:]
    _result = _result[:4] + [_result[4:]]
    _result = _BoostedTreesCalculateBestGainsPerFeatureOutput._make(_result)
    return _result