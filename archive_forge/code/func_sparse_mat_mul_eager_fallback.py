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
def sparse_mat_mul_eager_fallback(a: _atypes.TensorFuzzingAnnotation[TV_SparseMatMul_Ta], b: _atypes.TensorFuzzingAnnotation[TV_SparseMatMul_Tb], transpose_a: bool, transpose_b: bool, a_is_sparse: bool, b_is_sparse: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
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
    _attr_Ta, (a,) = _execute.args_to_matching_eager([a], ctx, [_dtypes.float32, _dtypes.bfloat16], _dtypes.float32)
    _attr_Tb, (b,) = _execute.args_to_matching_eager([b], ctx, [_dtypes.float32, _dtypes.bfloat16], _dtypes.float32)
    _inputs_flat = [a, b]
    _attrs = ('transpose_a', transpose_a, 'transpose_b', transpose_b, 'a_is_sparse', a_is_sparse, 'b_is_sparse', b_is_sparse, 'Ta', _attr_Ta, 'Tb', _attr_Tb)
    _result = _execute.execute(b'SparseMatMul', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseMatMul', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result