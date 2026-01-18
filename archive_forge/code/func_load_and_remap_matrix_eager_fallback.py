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
def load_and_remap_matrix_eager_fallback(ckpt_path: _atypes.TensorFuzzingAnnotation[_atypes.String], old_tensor_name: _atypes.TensorFuzzingAnnotation[_atypes.String], row_remapping: _atypes.TensorFuzzingAnnotation[_atypes.Int64], col_remapping: _atypes.TensorFuzzingAnnotation[_atypes.Int64], initializing_values: _atypes.TensorFuzzingAnnotation[_atypes.Float32], num_rows: int, num_cols: int, max_rows_in_memory: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    num_rows = _execute.make_int(num_rows, 'num_rows')
    num_cols = _execute.make_int(num_cols, 'num_cols')
    if max_rows_in_memory is None:
        max_rows_in_memory = -1
    max_rows_in_memory = _execute.make_int(max_rows_in_memory, 'max_rows_in_memory')
    ckpt_path = _ops.convert_to_tensor(ckpt_path, _dtypes.string)
    old_tensor_name = _ops.convert_to_tensor(old_tensor_name, _dtypes.string)
    row_remapping = _ops.convert_to_tensor(row_remapping, _dtypes.int64)
    col_remapping = _ops.convert_to_tensor(col_remapping, _dtypes.int64)
    initializing_values = _ops.convert_to_tensor(initializing_values, _dtypes.float32)
    _inputs_flat = [ckpt_path, old_tensor_name, row_remapping, col_remapping, initializing_values]
    _attrs = ('num_rows', num_rows, 'num_cols', num_cols, 'max_rows_in_memory', max_rows_in_memory)
    _result = _execute.execute(b'LoadAndRemapMatrix', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('LoadAndRemapMatrix', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result