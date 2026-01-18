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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export(v1=['train.sdca_shrink_l1'])
@deprecated_endpoints('train.sdca_shrink_l1')
def sdca_shrink_l1(weights: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], l1: float, l2: float, name=None):
    """Applies L1 regularization shrink step on the parameters.

  Args:
    weights: A list of `Tensor` objects with type mutable `float32`.
      a list of vectors where each value is the weight associated with a
      feature group.
    l1: A `float`. Symmetric l1 regularization strength.
    l2: A `float`.
      Symmetric l2 regularization strength. Should be a positive float.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("sdca_shrink_l1 op does not support eager execution. Arg 'weights' is a ref.")
    else:
        _result = _dispatcher_for_sdca_shrink_l1((weights, l1, l2, name), None)
        if _result is not NotImplemented:
            return _result
    if not isinstance(weights, (list, tuple)):
        raise TypeError("Expected list for 'weights' argument to 'sdca_shrink_l1' Op, not %r." % weights)
    _attr_num_features = len(weights)
    l1 = _execute.make_float(l1, 'l1')
    l2 = _execute.make_float(l2, 'l2')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('SdcaShrinkL1', weights=weights, l1=l1, l2=l2, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(sdca_shrink_l1, (), dict(weights=weights, l1=l1, l2=l2, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    return _op