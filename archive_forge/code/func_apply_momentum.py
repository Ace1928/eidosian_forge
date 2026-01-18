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
def apply_momentum(var: _atypes.TensorFuzzingAnnotation[TV_ApplyMomentum_T], accum: _atypes.TensorFuzzingAnnotation[TV_ApplyMomentum_T], lr: _atypes.TensorFuzzingAnnotation[TV_ApplyMomentum_T], grad: _atypes.TensorFuzzingAnnotation[TV_ApplyMomentum_T], momentum: _atypes.TensorFuzzingAnnotation[TV_ApplyMomentum_T], use_locking: bool=False, use_nesterov: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ApplyMomentum_T]:
    """Update '*var' according to the momentum scheme.

  Set use_nesterov = True if you want to use Nesterov momentum.

  accum = accum * momentum + grad
  var -= lr * accum

  Args:
    var: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
      Should be from a Variable().
    accum: A mutable `Tensor`. Must have the same type as `var`.
      Should be from a Variable().
    lr: A `Tensor`. Must have the same type as `var`.
      Scaling factor. Must be a scalar.
    grad: A `Tensor`. Must have the same type as `var`. The gradient.
    momentum: A `Tensor`. Must have the same type as `var`.
      Momentum. Must be a scalar.
    use_locking: An optional `bool`. Defaults to `False`.
      If `True`, updating of the var and accum tensors will be protected
      by a lock; otherwise the behavior is undefined, but may exhibit less
      contention.
    use_nesterov: An optional `bool`. Defaults to `False`.
      If `True`, the tensor passed to compute grad will be
      var - lr * momentum * accum, so in the end, the var you get is actually
      var - lr * momentum * accum.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `var`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("apply_momentum op does not support eager execution. Arg 'out' is a ref.")
    if use_locking is None:
        use_locking = False
    use_locking = _execute.make_bool(use_locking, 'use_locking')
    if use_nesterov is None:
        use_nesterov = False
    use_nesterov = _execute.make_bool(use_nesterov, 'use_nesterov')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ApplyMomentum', var=var, accum=accum, lr=lr, grad=grad, momentum=momentum, use_locking=use_locking, use_nesterov=use_nesterov, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'use_locking', _op._get_attr_bool('use_locking'), 'use_nesterov', _op._get_attr_bool('use_nesterov'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ApplyMomentum', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result