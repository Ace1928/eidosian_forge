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
def boosted_trees_create_quantile_stream_resource(quantile_stream_resource_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], epsilon: _atypes.TensorFuzzingAnnotation[_atypes.Float32], num_streams: _atypes.TensorFuzzingAnnotation[_atypes.Int64], max_elements: int=1099511627776, name=None):
    """Create the Resource for Quantile Streams.

  Args:
    quantile_stream_resource_handle: A `Tensor` of type `resource`.
      resource; Handle to quantile stream resource.
    epsilon: A `Tensor` of type `float32`.
      float; The required approximation error of the stream resource.
    num_streams: A `Tensor` of type `int64`.
      int; The number of streams managed by the resource that shares the same epsilon.
    max_elements: An optional `int`. Defaults to `1099511627776`.
      int; The maximum number of data points that can be fed to the stream.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BoostedTreesCreateQuantileStreamResource', name, quantile_stream_resource_handle, epsilon, num_streams, 'max_elements', max_elements)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return boosted_trees_create_quantile_stream_resource_eager_fallback(quantile_stream_resource_handle, epsilon, num_streams, max_elements=max_elements, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if max_elements is None:
        max_elements = 1099511627776
    max_elements = _execute.make_int(max_elements, 'max_elements')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BoostedTreesCreateQuantileStreamResource', quantile_stream_resource_handle=quantile_stream_resource_handle, epsilon=epsilon, num_streams=num_streams, max_elements=max_elements, name=name)
    return _op