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
def resource_apply_ada_max(var: _atypes.TensorFuzzingAnnotation[_atypes.Resource], m: _atypes.TensorFuzzingAnnotation[_atypes.Resource], v: _atypes.TensorFuzzingAnnotation[_atypes.Resource], beta1_power: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyAdaMax_T], lr: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyAdaMax_T], beta1: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyAdaMax_T], beta2: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyAdaMax_T], epsilon: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyAdaMax_T], grad: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyAdaMax_T], use_locking: bool=False, name=None):
    """Update '*var' according to the AdaMax algorithm.

  m_t <- beta1 * m_{t-1} + (1 - beta1) * g
  v_t <- max(beta2 * v_{t-1}, abs(g))
  variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    m: A `Tensor` of type `resource`. Should be from a Variable().
    v: A `Tensor` of type `resource`. Should be from a Variable().
    beta1_power: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Must be a scalar.
    lr: A `Tensor`. Must have the same type as `beta1_power`.
      Scaling factor. Must be a scalar.
    beta1: A `Tensor`. Must have the same type as `beta1_power`.
      Momentum factor. Must be a scalar.
    beta2: A `Tensor`. Must have the same type as `beta1_power`.
      Momentum factor. Must be a scalar.
    epsilon: A `Tensor`. Must have the same type as `beta1_power`.
      Ridge term. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `beta1_power`. The gradient.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var, m, and v tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResourceApplyAdaMax', name, var, m, v, beta1_power, lr, beta1, beta2, epsilon, grad, 'use_locking', use_locking)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resource_apply_ada_max_eager_fallback(var, m, v, beta1_power, lr, beta1, beta2, epsilon, grad, use_locking=use_locking, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if use_locking is None:
        use_locking = False
    use_locking = _execute.make_bool(use_locking, 'use_locking')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResourceApplyAdaMax', var=var, m=m, v=v, beta1_power=beta1_power, lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, grad=grad, use_locking=use_locking, name=name)
    return _op