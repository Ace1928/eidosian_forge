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
def sparse_mat_mul(a: _atypes.TensorFuzzingAnnotation[TV_SparseMatMul_Ta], b: _atypes.TensorFuzzingAnnotation[TV_SparseMatMul_Tb], transpose_a: bool=False, transpose_b: bool=False, a_is_sparse: bool=False, b_is_sparse: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Multiply matrix "a" by matrix "b".

  The inputs must be two-dimensional matrices and the inner dimension of "a" must
  match the outer dimension of "b". Both "a" and "b" must be `Tensor`s not
  `SparseTensor`s.  This op is optimized for the case where at least one of "a" or
  "b" is sparse, in the sense that they have a large proportion of zero values.
  The breakeven for using this versus a dense matrix multiply on one platform was
  30% zero values in the sparse matrix.

  The gradient computation of this operation will only take advantage of sparsity
  in the input gradient when that gradient comes from a Relu.

  Args:
    a: A `Tensor`. Must be one of the following types: `float32`, `bfloat16`.
    b: A `Tensor`. Must be one of the following types: `float32`, `bfloat16`.
    transpose_a: An optional `bool`. Defaults to `False`.
    transpose_b: An optional `bool`. Defaults to `False`.
    a_is_sparse: An optional `bool`. Defaults to `False`.
    b_is_sparse: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseMatMul', name, a, b, 'transpose_a', transpose_a, 'transpose_b', transpose_b, 'a_is_sparse', a_is_sparse, 'b_is_sparse', b_is_sparse)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_mat_mul_eager_fallback(a, b, transpose_a=transpose_a, transpose_b=transpose_b, a_is_sparse=a_is_sparse, b_is_sparse=b_is_sparse, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if transpose_a is None:
        transpose_a = False
    transpose_a = _execute.make_bool(transpose_a, 'transpose_a')
    if transpose_b is None:
        transpose_b = False
    transpose_b = _execute.make_bool(transpose_b, 'transpose_b')
    if a_is_sparse is None:
        a_is_sparse = False
    a_is_sparse = _execute.make_bool(a_is_sparse, 'a_is_sparse')
    if b_is_sparse is None:
        b_is_sparse = False
    b_is_sparse = _execute.make_bool(b_is_sparse, 'b_is_sparse')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseMatMul', a=a, b=b, transpose_a=transpose_a, transpose_b=transpose_b, a_is_sparse=a_is_sparse, b_is_sparse=b_is_sparse, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('transpose_a', _op._get_attr_bool('transpose_a'), 'transpose_b', _op._get_attr_bool('transpose_b'), 'a_is_sparse', _op._get_attr_bool('a_is_sparse'), 'b_is_sparse', _op._get_attr_bool('b_is_sparse'), 'Ta', _op._get_attr_type('Ta'), 'Tb', _op._get_attr_type('Tb'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseMatMul', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result