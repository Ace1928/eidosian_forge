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
def mat_mul(a: _atypes.TensorFuzzingAnnotation[TV_MatMul_T], b: _atypes.TensorFuzzingAnnotation[TV_MatMul_T], transpose_a: bool=False, transpose_b: bool=False, grad_a: bool=False, grad_b: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_MatMul_T]:
    """Multiply the matrix "a" by the matrix "b".

  The inputs must be two-dimensional matrices and the inner dimension of
  "a" (after being transposed if transpose_a is true) must match the
  outer dimension of "b" (after being transposed if transposed_b is
  true).

  *Note*: The default kernel implementation for MatMul on GPUs uses
  cublas.

  Args:
    a: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `complex64`, `complex128`.
    b: A `Tensor`. Must have the same type as `a`.
    transpose_a: An optional `bool`. Defaults to `False`.
      If true, "a" is transposed before multiplication.
    transpose_b: An optional `bool`. Defaults to `False`.
      If true, "b" is transposed before multiplication.
    grad_a: An optional `bool`. Defaults to `False`.
    grad_b: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MatMul', name, a, b, 'transpose_a', transpose_a, 'transpose_b', transpose_b, 'grad_a', grad_a, 'grad_b', grad_b)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return mat_mul_eager_fallback(a, b, transpose_a=transpose_a, transpose_b=transpose_b, grad_a=grad_a, grad_b=grad_b, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if transpose_a is None:
        transpose_a = False
    transpose_a = _execute.make_bool(transpose_a, 'transpose_a')
    if transpose_b is None:
        transpose_b = False
    transpose_b = _execute.make_bool(transpose_b, 'transpose_b')
    if grad_a is None:
        grad_a = False
    grad_a = _execute.make_bool(grad_a, 'grad_a')
    if grad_b is None:
        grad_b = False
    grad_b = _execute.make_bool(grad_b, 'grad_b')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MatMul', a=a, b=b, transpose_a=transpose_a, transpose_b=transpose_b, grad_a=grad_a, grad_b=grad_b, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('transpose_a', _op._get_attr_bool('transpose_a'), 'transpose_b', _op._get_attr_bool('transpose_b'), 'T', _op._get_attr_type('T'), 'grad_a', _op._get_attr_bool('grad_a'), 'grad_b', _op._get_attr_bool('grad_b'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MatMul', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result