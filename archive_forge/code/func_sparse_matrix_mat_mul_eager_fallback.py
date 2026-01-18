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
def sparse_matrix_mat_mul_eager_fallback(a: _atypes.TensorFuzzingAnnotation[_atypes.Variant], b: _atypes.TensorFuzzingAnnotation[TV_SparseMatrixMatMul_T], transpose_a: bool, transpose_b: bool, adjoint_a: bool, adjoint_b: bool, transpose_output: bool, conjugate_output: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_SparseMatrixMatMul_T]:
    if transpose_a is None:
        transpose_a = False
    transpose_a = _execute.make_bool(transpose_a, 'transpose_a')
    if transpose_b is None:
        transpose_b = False
    transpose_b = _execute.make_bool(transpose_b, 'transpose_b')
    if adjoint_a is None:
        adjoint_a = False
    adjoint_a = _execute.make_bool(adjoint_a, 'adjoint_a')
    if adjoint_b is None:
        adjoint_b = False
    adjoint_b = _execute.make_bool(adjoint_b, 'adjoint_b')
    if transpose_output is None:
        transpose_output = False
    transpose_output = _execute.make_bool(transpose_output, 'transpose_output')
    if conjugate_output is None:
        conjugate_output = False
    conjugate_output = _execute.make_bool(conjugate_output, 'conjugate_output')
    _attr_T, (b,) = _execute.args_to_matching_eager([b], ctx, [])
    a = _ops.convert_to_tensor(a, _dtypes.variant)
    _inputs_flat = [a, b]
    _attrs = ('T', _attr_T, 'transpose_a', transpose_a, 'transpose_b', transpose_b, 'adjoint_a', adjoint_a, 'adjoint_b', adjoint_b, 'transpose_output', transpose_output, 'conjugate_output', conjugate_output)
    _result = _execute.execute(b'SparseMatrixMatMul', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseMatrixMatMul', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result