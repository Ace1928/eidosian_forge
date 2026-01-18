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
def reduce_join_eager_fallback(inputs: _atypes.TensorFuzzingAnnotation[_atypes.String], reduction_indices: _atypes.TensorFuzzingAnnotation[_atypes.Int32], keep_dims: bool, separator: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    if keep_dims is None:
        keep_dims = False
    keep_dims = _execute.make_bool(keep_dims, 'keep_dims')
    if separator is None:
        separator = ''
    separator = _execute.make_str(separator, 'separator')
    inputs = _ops.convert_to_tensor(inputs, _dtypes.string)
    reduction_indices = _ops.convert_to_tensor(reduction_indices, _dtypes.int32)
    _inputs_flat = [inputs, reduction_indices]
    _attrs = ('keep_dims', keep_dims, 'separator', separator)
    _result = _execute.execute(b'ReduceJoin', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ReduceJoin', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result