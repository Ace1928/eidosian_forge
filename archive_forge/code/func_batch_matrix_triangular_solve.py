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
def batch_matrix_triangular_solve(matrix: _atypes.TensorFuzzingAnnotation[TV_BatchMatrixTriangularSolve_T], rhs: _atypes.TensorFuzzingAnnotation[TV_BatchMatrixTriangularSolve_T], lower: bool=True, adjoint: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_BatchMatrixTriangularSolve_T]:
    """TODO: add doc.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
    lower: An optional `bool`. Defaults to `True`.
    adjoint: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BatchMatrixTriangularSolve', name, matrix, rhs, 'lower', lower, 'adjoint', adjoint)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return batch_matrix_triangular_solve_eager_fallback(matrix, rhs, lower=lower, adjoint=adjoint, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if lower is None:
        lower = True
    lower = _execute.make_bool(lower, 'lower')
    if adjoint is None:
        adjoint = False
    adjoint = _execute.make_bool(adjoint, 'adjoint')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BatchMatrixTriangularSolve', matrix=matrix, rhs=rhs, lower=lower, adjoint=adjoint, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('lower', _op._get_attr_bool('lower'), 'adjoint', _op._get_attr_bool('adjoint'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BatchMatrixTriangularSolve', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result