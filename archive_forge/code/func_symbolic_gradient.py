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
def symbolic_gradient(input, Tout, f, name=None):
    """Computes the gradient function for function f via backpropagation.

  Args:
    input: A list of `Tensor` objects. a list of input tensors of size N + M;
    Tout: A list of `tf.DTypes` that has length `>= 1`.
      the type list for the input list.
    f: A function decorated with @Defun.
      The function we want to compute the gradient for.

      The function 'f' must be a numerical function which takes N inputs and
      produces M outputs. Its gradient function 'g', which is computed by
      this SymbolicGradient op is a function taking N + M inputs and
      produces N outputs.

      I.e. if we have
         (y1, y2, ..., y_M) = f(x1, x2, ..., x_N),
      then, g is
         (dL/dx1, dL/dx2, ..., dL/dx_N) = g(x1, x2, ..., x_N,
                                           dL/dy1, dL/dy2, ..., dL/dy_M),

      where L is a scalar-value function of (x1, x2, ..., xN) (e.g., the
      loss function). dL/dx_i is the partial derivative of L with respect
      to x_i.

      (Needs some math expert to say the comment above better.)
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SymbolicGradient', name, input, 'Tout', Tout, 'f', f)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return symbolic_gradient_eager_fallback(input, Tout=Tout, f=f, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'symbolic_gradient' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SymbolicGradient', input=input, Tout=Tout, f=f, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tin', _op.get_attr('Tin'), 'Tout', _op.get_attr('Tout'), 'f', _op.get_attr('f'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SymbolicGradient', _inputs_flat, _attrs, _result)
    return _result