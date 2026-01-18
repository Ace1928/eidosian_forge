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
def sparse_split_eager_fallback(split_dim: _atypes.TensorFuzzingAnnotation[_atypes.Int64], indices: _atypes.TensorFuzzingAnnotation[_atypes.Int64], values: _atypes.TensorFuzzingAnnotation[TV_SparseSplit_T], shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], num_split: int, name, ctx):
    num_split = _execute.make_int(num_split, 'num_split')
    _attr_T, (values,) = _execute.args_to_matching_eager([values], ctx, [])
    split_dim = _ops.convert_to_tensor(split_dim, _dtypes.int64)
    indices = _ops.convert_to_tensor(indices, _dtypes.int64)
    shape = _ops.convert_to_tensor(shape, _dtypes.int64)
    _inputs_flat = [split_dim, indices, values, shape]
    _attrs = ('num_split', num_split, 'T', _attr_T)
    _result = _execute.execute(b'SparseSplit', num_split + num_split + num_split, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseSplit', _inputs_flat, _attrs, _result)
    _result = [_result[:num_split]] + _result[num_split:]
    _result = _result[:1] + [_result[1:1 + num_split]] + _result[1 + num_split:]
    _result = _result[:2] + [_result[2:]]
    _result = _SparseSplitOutput._make(_result)
    return _result