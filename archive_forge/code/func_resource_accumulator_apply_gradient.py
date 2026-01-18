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
def resource_accumulator_apply_gradient(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], local_step: _atypes.TensorFuzzingAnnotation[_atypes.Int64], gradient: _atypes.TensorFuzzingAnnotation[TV_ResourceAccumulatorApplyGradient_dtype], name=None):
    """Applies a gradient to a given accumulator.

  Does not add if local_step is lesser than the accumulator's global_step.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a accumulator.
    local_step: A `Tensor` of type `int64`.
      The local_step value at which the gradient was computed.
    gradient: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      A tensor of the gradient to be accumulated.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResourceAccumulatorApplyGradient', name, handle, local_step, gradient)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resource_accumulator_apply_gradient_eager_fallback(handle, local_step, gradient, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResourceAccumulatorApplyGradient', handle=handle, local_step=local_step, gradient=gradient, name=name)
    return _op