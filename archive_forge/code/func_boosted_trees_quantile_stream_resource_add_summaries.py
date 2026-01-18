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
def boosted_trees_quantile_stream_resource_add_summaries(quantile_stream_resource_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], summaries: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], name=None):
    """Add the quantile summaries to each quantile stream resource.

  An op that adds a list of quantile summaries to a quantile stream resource. Each
  summary Tensor is rank 2, containing summaries (value, weight, min_rank, max_rank)
  for a single feature.

  Args:
    quantile_stream_resource_handle: A `Tensor` of type `resource`.
      resource handle referring to a QuantileStreamResource.
    summaries: A list of `Tensor` objects with type `float32`.
      string; List of Rank 2 Tensor each containing the summaries for a single feature.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BoostedTreesQuantileStreamResourceAddSummaries', name, quantile_stream_resource_handle, summaries)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return boosted_trees_quantile_stream_resource_add_summaries_eager_fallback(quantile_stream_resource_handle, summaries, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(summaries, (list, tuple)):
        raise TypeError("Expected list for 'summaries' argument to 'boosted_trees_quantile_stream_resource_add_summaries' Op, not %r." % summaries)
    _attr_num_features = len(summaries)
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BoostedTreesQuantileStreamResourceAddSummaries', quantile_stream_resource_handle=quantile_stream_resource_handle, summaries=summaries, name=name)
    return _op