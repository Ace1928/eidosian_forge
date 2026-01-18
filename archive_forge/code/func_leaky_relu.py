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
def leaky_relu(features: _atypes.TensorFuzzingAnnotation[TV_LeakyRelu_T], alpha: float=0.2, name=None) -> _atypes.TensorFuzzingAnnotation[TV_LeakyRelu_T]:
    """Computes rectified linear: `max(features, features * alpha)`.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
    alpha: An optional `float`. Defaults to `0.2`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `features`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'LeakyRelu', name, features, 'alpha', alpha)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return leaky_relu_eager_fallback(features, alpha=alpha, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if alpha is None:
        alpha = 0.2
    alpha = _execute.make_float(alpha, 'alpha')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('LeakyRelu', features=features, alpha=alpha, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('alpha', _op.get_attr('alpha'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('LeakyRelu', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result