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
def sparse_tensor_dense_mat_mul(a_indices: _atypes.TensorFuzzingAnnotation[TV_SparseTensorDenseMatMul_Tindices], a_values: _atypes.TensorFuzzingAnnotation[TV_SparseTensorDenseMatMul_T], a_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], b: _atypes.TensorFuzzingAnnotation[TV_SparseTensorDenseMatMul_T], adjoint_a: bool=False, adjoint_b: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_SparseTensorDenseMatMul_T]:
    """Multiply SparseTensor (of rank 2) "A" by dense matrix "B".

  No validity checking is performed on the indices of A.  However, the following
  input format is recommended for optimal behavior:

  if adjoint_a == false:
    A should be sorted in lexicographically increasing order.  Use SparseReorder
    if you're not sure.
  if adjoint_a == true:
    A should be sorted in order of increasing dimension 1 (i.e., "column major"
    order instead of "row major" order).

  Args:
    a_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2-D.  The `indices` of the `SparseTensor`, size `[nnz, 2]` Matrix.
    a_values: A `Tensor`.
      1-D.  The `values` of the `SparseTensor`, size `[nnz]` Vector.
    a_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the `SparseTensor`, size `[2]` Vector.
    b: A `Tensor`. Must have the same type as `a_values`.
      2-D.  A dense Matrix.
    adjoint_a: An optional `bool`. Defaults to `False`.
      Use the adjoint of A in the matrix multiply.  If A is complex, this
      is transpose(conj(A)).  Otherwise it's transpose(A).
    adjoint_b: An optional `bool`. Defaults to `False`.
      Use the adjoint of B in the matrix multiply.  If B is complex, this
      is transpose(conj(B)).  Otherwise it's transpose(B).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `a_values`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseTensorDenseMatMul', name, a_indices, a_values, a_shape, b, 'adjoint_a', adjoint_a, 'adjoint_b', adjoint_b)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_tensor_dense_mat_mul_eager_fallback(a_indices, a_values, a_shape, b, adjoint_a=adjoint_a, adjoint_b=adjoint_b, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if adjoint_a is None:
        adjoint_a = False
    adjoint_a = _execute.make_bool(adjoint_a, 'adjoint_a')
    if adjoint_b is None:
        adjoint_b = False
    adjoint_b = _execute.make_bool(adjoint_b, 'adjoint_b')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseTensorDenseMatMul', a_indices=a_indices, a_values=a_values, a_shape=a_shape, b=b, adjoint_a=adjoint_a, adjoint_b=adjoint_b, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tindices', _op._get_attr_type('Tindices'), 'adjoint_a', _op._get_attr_bool('adjoint_a'), 'adjoint_b', _op._get_attr_bool('adjoint_b'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseTensorDenseMatMul', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result