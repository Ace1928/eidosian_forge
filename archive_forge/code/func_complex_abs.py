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
def complex_abs(x: _atypes.TensorFuzzingAnnotation[TV_ComplexAbs_T], Tout: TV_ComplexAbs_Tout=_dtypes.float32, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ComplexAbs_Tout]:
    """Computes the complex absolute value of a tensor.

  Given a tensor `x` of complex numbers, this operation returns a tensor of type
  `float` or `double` that is the absolute value of each element in `x`. All
  elements in `x` must be complex numbers of the form \\\\(a + bj\\\\). The absolute
  value is computed as \\\\( \\sqrt{a^2 + b^2}\\\\).

  For example:

  >>> x = tf.complex(3.0, 4.0)
  >>> print((tf.raw_ops.ComplexAbs(x=x, Tout=tf.dtypes.float32, name=None)).numpy())
  5.0

  Args:
    x: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
    Tout: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ComplexAbs', name, x, 'Tout', Tout)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return complex_abs_eager_fallback(x, Tout=Tout, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if Tout is None:
        Tout = _dtypes.float32
    Tout = _execute.make_type(Tout, 'Tout')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ComplexAbs', x=x, Tout=Tout, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tout', _op._get_attr_type('Tout'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ComplexAbs', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result