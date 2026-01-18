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
def sparse_slice_eager_fallback(indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], values: _atypes.TensorFuzzingAnnotation[TV_SparseSlice_T], shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], start: _atypes.TensorFuzzingAnnotation[_atypes.Int64], size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name, ctx):
    _attr_T, (values,) = _execute.args_to_matching_eager([values], ctx, [])
    indices = _ops.convert_to_tensor(indices, _dtypes.int64)
    shape = _ops.convert_to_tensor(shape, _dtypes.int64)
    start = _ops.convert_to_tensor(start, _dtypes.int64)
    size = _ops.convert_to_tensor(size, _dtypes.int64)
    _inputs_flat = [indices, values, shape, start, size]
    _attrs = ('T', _attr_T)
    _result = _execute.execute(b'SparseSlice', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseSlice', _inputs_flat, _attrs, _result)
    _result = _SparseSliceOutput._make(_result)
    return _result