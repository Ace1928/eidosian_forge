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
def is_boosted_trees_quantile_stream_resource_initialized(quantile_stream_resource_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Bool]:
    """Checks whether a quantile stream has been initialized.

  An Op that checks if quantile stream resource is initialized.

  Args:
    quantile_stream_resource_handle: A `Tensor` of type `resource`.
      resource; The reference to quantile stream resource handle.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'IsBoostedTreesQuantileStreamResourceInitialized', name, quantile_stream_resource_handle)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return is_boosted_trees_quantile_stream_resource_initialized_eager_fallback(quantile_stream_resource_handle, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('IsBoostedTreesQuantileStreamResourceInitialized', quantile_stream_resource_handle=quantile_stream_resource_handle, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('IsBoostedTreesQuantileStreamResourceInitialized', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result