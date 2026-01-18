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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('linalg.sqrtm', 'matrix_square_root')
def matrix_square_root(input: _atypes.TensorFuzzingAnnotation[TV_MatrixSquareRoot_T], name=None) -> _atypes.TensorFuzzingAnnotation[TV_MatrixSquareRoot_T]:
    """Computes the matrix square root of one or more square matrices:

  matmul(sqrtm(A), sqrtm(A)) = A

  The input matrix should be invertible. If the input matrix is real, it should
  have no eigenvalues which are real and negative (pairs of complex conjugate
  eigenvalues are allowed).

  The matrix square root is computed by first reducing the matrix to
  quasi-triangular form with the real Schur decomposition. The square root
  of the quasi-triangular matrix is then computed directly. Details of
  the algorithm can be found in: Nicholas J. Higham, "Computing real
  square roots of a real matrix", Linear Algebra Appl., 1987.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a tensor of the same shape as the input
  containing the matrix square root for all input submatrices `[..., :, :]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`, `half`, `complex64`, `complex128`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MatrixSquareRoot', name, input)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_matrix_square_root((input, name), None)
            if _result is not NotImplemented:
                return _result
            return matrix_square_root_eager_fallback(input, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(matrix_square_root, (), dict(input=input, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_matrix_square_root((input, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('MatrixSquareRoot', input=input, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(matrix_square_root, (), dict(input=input, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MatrixSquareRoot', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result