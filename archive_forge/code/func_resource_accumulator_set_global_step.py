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
def resource_accumulator_set_global_step(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], new_global_step: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name=None):
    """Updates the accumulator with a new value for global_step.

  Logs warning if the accumulator's value is already higher than
  new_global_step.

  Args:
    handle: A `Tensor` of type `resource`. The handle to an accumulator.
    new_global_step: A `Tensor` of type `int64`.
      The new global_step value to set.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResourceAccumulatorSetGlobalStep', name, handle, new_global_step)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resource_accumulator_set_global_step_eager_fallback(handle, new_global_step, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResourceAccumulatorSetGlobalStep', handle=handle, new_global_step=new_global_step, name=name)
    return _op