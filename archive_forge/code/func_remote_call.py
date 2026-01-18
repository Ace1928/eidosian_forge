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
def remote_call(target: _atypes.TensorFuzzingAnnotation[_atypes.String], args, Tout, f, name=None):
    """Runs function `f` on a remote device indicated by `target`.

  Args:
    target: A `Tensor` of type `string`.
      A fully specified device name where we want to run the function.
    args: A list of `Tensor` objects. A list of arguments for the function.
    Tout: A list of `tf.DTypes` that has length `>= 1`.
      The type list for the return values.
    f: A function decorated with @Defun. The function to run remotely.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RemoteCall', name, target, args, 'Tout', Tout, 'f', f)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return remote_call_eager_fallback(target, args, Tout=Tout, f=f, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'remote_call' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RemoteCall', target=target, args=args, Tout=Tout, f=f, name=name)
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('Tin', _op.get_attr('Tin'), 'Tout', _op.get_attr('Tout'), 'f', _op.get_attr('f'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RemoteCall', _inputs_flat, _attrs, _result)
    return _result