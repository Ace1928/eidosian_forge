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
def matrix_solve_ls(matrix: _atypes.TensorFuzzingAnnotation[TV_MatrixSolveLs_T], rhs: _atypes.TensorFuzzingAnnotation[TV_MatrixSolveLs_T], l2_regularizer: _atypes.TensorFuzzingAnnotation[_atypes.Float64], fast: bool=True, name=None) -> _atypes.TensorFuzzingAnnotation[TV_MatrixSolveLs_T]:
    """Solves one or more linear least-squares problems.

  `matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
  form real or complex matrices of size `[M, N]`. `Rhs` is a tensor of the same
  type as `matrix` and shape `[..., M, K]`.
  The output is a tensor shape `[..., N, K]` where each output matrix solves
  each of the equations
  `matrix[..., :, :]` * `output[..., :, :]` = `rhs[..., :, :]`
  in the least squares sense.

  We use the following notation for (complex) matrix and right-hand sides
  in the batch:

  `matrix`=\\\\(A \\in \\mathbb{C}^{m \\times n}\\\\),
  `rhs`=\\\\(B  \\in \\mathbb{C}^{m \\times k}\\\\),
  `output`=\\\\(X  \\in \\mathbb{C}^{n \\times k}\\\\),
  `l2_regularizer`=\\\\(\\lambda \\in \\mathbb{R}\\\\).

  If `fast` is `True`, then the solution is computed by solving the normal
  equations using Cholesky decomposition. Specifically, if \\\\(m \\ge n\\\\) then
  \\\\(X = (A^H A + \\lambda I)^{-1} A^H B\\\\), which solves the least-squares
  problem \\\\(X = \\mathrm{argmin}_{Z \\in \\Re^{n \\times k} } ||A Z - B||_F^2 + \\lambda ||Z||_F^2\\\\).
  If \\\\(m \\lt n\\\\) then `output` is computed as
  \\\\(X = A^H (A A^H + \\lambda I)^{-1} B\\\\), which (for \\\\(\\lambda = 0\\\\)) is the
  minimum-norm solution to the under-determined linear system, i.e.
  \\\\(X = \\mathrm{argmin}_{Z \\in \\mathbb{C}^{n \\times k} } ||Z||_F^2 \\\\),
  subject to \\\\(A Z = B\\\\). Notice that the fast path is only numerically stable
  when \\\\(A\\\\) is numerically full rank and has a condition number
  \\\\(\\mathrm{cond}(A) \\lt \\frac{1}{\\sqrt{\\epsilon_{mach} } }\\\\) or \\\\(\\lambda\\\\) is
  sufficiently large.

  If `fast` is `False` an algorithm based on the numerically robust complete
  orthogonal decomposition is used. This computes the minimum-norm
  least-squares solution, even when \\\\(A\\\\) is rank deficient. This path is
  typically 6-7 times slower than the fast path. If `fast` is `False` then
  `l2_regularizer` is ignored.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`, `half`, `complex64`, `complex128`.
      Shape is `[..., M, N]`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
      Shape is `[..., M, K]`.
    l2_regularizer: A `Tensor` of type `float64`. Scalar tensor.

      @compatibility(numpy)
      Equivalent to np.linalg.lstsq
      @end_compatibility
    fast: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MatrixSolveLs', name, matrix, rhs, l2_regularizer, 'fast', fast)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return matrix_solve_ls_eager_fallback(matrix, rhs, l2_regularizer, fast=fast, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if fast is None:
        fast = True
    fast = _execute.make_bool(fast, 'fast')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MatrixSolveLs', matrix=matrix, rhs=rhs, l2_regularizer=l2_regularizer, fast=fast, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'fast', _op._get_attr_bool('fast'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MatrixSolveLs', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result