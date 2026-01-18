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
def sparse_to_sparse_set_operation_eager_fallback(set1_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], set1_values: _atypes.TensorFuzzingAnnotation[TV_SparseToSparseSetOperation_T], set1_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], set2_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], set2_values: _atypes.TensorFuzzingAnnotation[TV_SparseToSparseSetOperation_T], set2_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], set_operation: str, validate_indices: bool, name, ctx):
    set_operation = _execute.make_str(set_operation, 'set_operation')
    if validate_indices is None:
        validate_indices = True
    validate_indices = _execute.make_bool(validate_indices, 'validate_indices')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([set1_values, set2_values], ctx, [_dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.string])
    set1_values, set2_values = _inputs_T
    set1_indices = _ops.convert_to_tensor(set1_indices, _dtypes.int64)
    set1_shape = _ops.convert_to_tensor(set1_shape, _dtypes.int64)
    set2_indices = _ops.convert_to_tensor(set2_indices, _dtypes.int64)
    set2_shape = _ops.convert_to_tensor(set2_shape, _dtypes.int64)
    _inputs_flat = [set1_indices, set1_values, set1_shape, set2_indices, set2_values, set2_shape]
    _attrs = ('set_operation', set_operation, 'validate_indices', validate_indices, 'T', _attr_T)
    _result = _execute.execute(b'SparseToSparseSetOperation', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseToSparseSetOperation', _inputs_flat, _attrs, _result)
    _result = _SparseToSparseSetOperationOutput._make(_result)
    return _result