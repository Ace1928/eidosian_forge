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
def batch_mat_mul_v3(x: _atypes.TensorFuzzingAnnotation[TV_BatchMatMulV3_Ta], y: _atypes.TensorFuzzingAnnotation[TV_BatchMatMulV3_Tb], Tout: TV_BatchMatMulV3_Tout, adj_x: bool=False, adj_y: bool=False, grad_x: bool=False, grad_y: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_BatchMatMulV3_Tout]:
    """Multiplies slices of two tensors in batches.

  Multiplies all slices of `Tensor` `x` and `y` (each slice can be
  viewed as an element of a batch), and arranges the individual results
  in a single output tensor of the same batch size. Each of the
  individual slices can optionally be adjointed (to adjoint a matrix
  means to transpose and conjugate it) before multiplication by setting
  the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

  The input tensors `x` and `y` are 2-D or higher with shape `[..., r_x, c_x]`
  and `[..., r_y, c_y]`.

  The output tensor is 2-D or higher with shape `[..., r_o, c_o]`, where:

      r_o = c_x if adj_x else r_x
      c_o = r_y if adj_y else c_y

  It is computed as:

      output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

  *NOTE*: `BatchMatMulV3` supports broadcasting in the batch dimensions. More
  about broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

  Args:
    x: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
      2-D or higher with shape `[..., r_x, c_x]`.
    y: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
      2-D or higher with shape `[..., r_y, c_y]`.
    Tout: A `tf.DType` from: `tf.bfloat16, tf.half, tf.float32, tf.float64, tf.int16, tf.int32, tf.int64, tf.complex64, tf.complex128`.
      If not spcified, Tout is the same type to input type.
    adj_x: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `x`. Defaults to `False`.
    adj_y: An optional `bool`. Defaults to `False`.
      If `True`, adjoint the slices of `y`. Defaults to `False`.
    grad_x: An optional `bool`. Defaults to `False`.
    grad_y: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BatchMatMulV3', name, x, y, 'Tout', Tout, 'adj_x', adj_x, 'adj_y', adj_y, 'grad_x', grad_x, 'grad_y', grad_y)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return batch_mat_mul_v3_eager_fallback(x, y, Tout=Tout, adj_x=adj_x, adj_y=adj_y, grad_x=grad_x, grad_y=grad_y, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    Tout = _execute.make_type(Tout, 'Tout')
    if adj_x is None:
        adj_x = False
    adj_x = _execute.make_bool(adj_x, 'adj_x')
    if adj_y is None:
        adj_y = False
    adj_y = _execute.make_bool(adj_y, 'adj_y')
    if grad_x is None:
        grad_x = False
    grad_x = _execute.make_bool(grad_x, 'grad_x')
    if grad_y is None:
        grad_y = False
    grad_y = _execute.make_bool(grad_y, 'grad_y')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BatchMatMulV3', x=x, y=y, Tout=Tout, adj_x=adj_x, adj_y=adj_y, grad_x=grad_x, grad_y=grad_y, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Ta', _op._get_attr_type('Ta'), 'Tb', _op._get_attr_type('Tb'), 'Tout', _op._get_attr_type('Tout'), 'adj_x', _op._get_attr_bool('adj_x'), 'adj_y', _op._get_attr_bool('adj_y'), 'grad_x', _op._get_attr_bool('grad_x'), 'grad_y', _op._get_attr_bool('grad_y'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BatchMatMulV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result