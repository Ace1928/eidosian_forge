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
def boosted_trees_predict(tree_ensemble_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], bucketized_features: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], logits_dimension: int, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Runs multiple additive regression ensemble predictors on input instances and

  computes the logits. It is designed to be used during prediction.
  It traverses all the trees and calculates the final score for each instance.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
    bucketized_features: A list of at least 1 `Tensor` objects with type `int32`.
      A list of rank 1 Tensors containing bucket id for each
      feature.
    logits_dimension: An `int`.
      scalar, dimension of the logits, to be used for partial logits
      shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BoostedTreesPredict', name, tree_ensemble_handle, bucketized_features, 'logits_dimension', logits_dimension)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return boosted_trees_predict_eager_fallback(tree_ensemble_handle, bucketized_features, logits_dimension=logits_dimension, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(bucketized_features, (list, tuple)):
        raise TypeError("Expected list for 'bucketized_features' argument to 'boosted_trees_predict' Op, not %r." % bucketized_features)
    _attr_num_bucketized_features = len(bucketized_features)
    logits_dimension = _execute.make_int(logits_dimension, 'logits_dimension')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BoostedTreesPredict', tree_ensemble_handle=tree_ensemble_handle, bucketized_features=bucketized_features, logits_dimension=logits_dimension, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('num_bucketized_features', _op._get_attr_int('num_bucketized_features'), 'logits_dimension', _op._get_attr_int('logits_dimension'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BoostedTreesPredict', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result