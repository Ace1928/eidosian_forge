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
def sparse_matrix_add_eager_fallback(a: _atypes.TensorFuzzingAnnotation[_atypes.Variant], b: _atypes.TensorFuzzingAnnotation[_atypes.Variant], alpha: _atypes.TensorFuzzingAnnotation[TV_SparseMatrixAdd_T], beta: _atypes.TensorFuzzingAnnotation[TV_SparseMatrixAdd_T], name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([alpha, beta], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.complex64, _dtypes.complex128])
    alpha, beta = _inputs_T
    a = _ops.convert_to_tensor(a, _dtypes.variant)
    b = _ops.convert_to_tensor(b, _dtypes.variant)
    _inputs_flat = [a, b, alpha, beta]
    _attrs = ('T', _attr_T)
    _result = _execute.execute(b'SparseMatrixAdd', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseMatrixAdd', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result