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
def boosted_trees_make_quantile_summaries(float_values: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], example_weights: _atypes.TensorFuzzingAnnotation[_atypes.Float32], epsilon: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name=None):
    """Makes the summary of quantiles for the batch.

  An op that takes a list of tensors (one tensor per feature) and outputs the
  quantile summaries for each tensor.

  Args:
    float_values: A list of `Tensor` objects with type `float32`.
      float; List of Rank 1 Tensors each containing values for a single feature.
    example_weights: A `Tensor` of type `float32`.
      float; Rank 1 Tensor with weights per instance.
    epsilon: A `Tensor` of type `float32`.
      float; The required maximum approximation error.
    name: A name for the operation (optional).

  Returns:
    A list with the same length as `float_values` of `Tensor` objects with type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BoostedTreesMakeQuantileSummaries', name, float_values, example_weights, epsilon)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return boosted_trees_make_quantile_summaries_eager_fallback(float_values, example_weights, epsilon, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(float_values, (list, tuple)):
        raise TypeError("Expected list for 'float_values' argument to 'boosted_trees_make_quantile_summaries' Op, not %r." % float_values)
    _attr_num_features = len(float_values)
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BoostedTreesMakeQuantileSummaries', float_values=float_values, example_weights=example_weights, epsilon=epsilon, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('num_features', _op._get_attr_int('num_features'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BoostedTreesMakeQuantileSummaries', _inputs_flat, _attrs, _result)
    return _result