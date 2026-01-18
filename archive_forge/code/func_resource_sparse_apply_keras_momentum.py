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
def resource_sparse_apply_keras_momentum(var: _atypes.TensorFuzzingAnnotation[_atypes.Resource], accum: _atypes.TensorFuzzingAnnotation[_atypes.Resource], lr: _atypes.TensorFuzzingAnnotation[TV_ResourceSparseApplyKerasMomentum_T], grad: _atypes.TensorFuzzingAnnotation[TV_ResourceSparseApplyKerasMomentum_T], indices: _atypes.TensorFuzzingAnnotation[TV_ResourceSparseApplyKerasMomentum_Tindices], momentum: _atypes.TensorFuzzingAnnotation[TV_ResourceSparseApplyKerasMomentum_T], use_locking: bool=False, use_nesterov: bool=False, name=None):
    """Update relevant entries in '*var' and '*accum' according to the momentum scheme.

  Set use_nesterov = True if you want to use Nesterov momentum.

  That is for rows we have grad for, we update var and accum as follows:

  accum = accum * momentum - lr * grad
  var += accum

  Args:
    var: A `Tensor` of type `resource`. Should be from a Variable().
    accum: A `Tensor` of type `resource`. Should be from a Variable().
    lr: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Learning rate. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `lr`. The gradient.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A vector of indices into the first dimension of var and accum.
    momentum: A `Tensor`. Must have the same type as `lr`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, the tensor passed to compute grad will be
      var + momentum * accum, so in the end, the var you get is actually
      var + momentum * accum.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResourceSparseApplyKerasMomentum', name, var, accum, lr, grad, indices, momentum, 'use_locking', use_locking, 'use_nesterov', use_nesterov)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resource_sparse_apply_keras_momentum_eager_fallback(var, accum, lr, grad, indices, momentum, use_locking=use_locking, use_nesterov=use_nesterov, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if use_locking is None:
        use_locking = False
    use_locking = _execute.make_bool(use_locking, 'use_locking')
    if use_nesterov is None:
        use_nesterov = False
    use_nesterov = _execute.make_bool(use_nesterov, 'use_nesterov')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResourceSparseApplyKerasMomentum', var=var, accum=accum, lr=lr, grad=grad, indices=indices, momentum=momentum, use_locking=use_locking, use_nesterov=use_nesterov, name=name)
    return _op