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
def fractional_avg_pool_grad_eager_fallback(orig_input_tensor_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int64], out_backprop: _atypes.TensorFuzzingAnnotation[TV_FractionalAvgPoolGrad_T], row_pooling_sequence: _atypes.TensorFuzzingAnnotation[_atypes.Int64], col_pooling_sequence: _atypes.TensorFuzzingAnnotation[_atypes.Int64], overlapping: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_FractionalAvgPoolGrad_T]:
    if overlapping is None:
        overlapping = False
    overlapping = _execute.make_bool(overlapping, 'overlapping')
    _attr_T, (out_backprop,) = _execute.args_to_matching_eager([out_backprop], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64])
    orig_input_tensor_shape = _ops.convert_to_tensor(orig_input_tensor_shape, _dtypes.int64)
    row_pooling_sequence = _ops.convert_to_tensor(row_pooling_sequence, _dtypes.int64)
    col_pooling_sequence = _ops.convert_to_tensor(col_pooling_sequence, _dtypes.int64)
    _inputs_flat = [orig_input_tensor_shape, out_backprop, row_pooling_sequence, col_pooling_sequence]
    _attrs = ('overlapping', overlapping, 'T', _attr_T)
    _result = _execute.execute(b'FractionalAvgPoolGrad', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('FractionalAvgPoolGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result