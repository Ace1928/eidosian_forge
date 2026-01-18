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
def parameterized_truncated_normal_eager_fallback(shape: _atypes.TensorFuzzingAnnotation[TV_ParameterizedTruncatedNormal_T], means: _atypes.TensorFuzzingAnnotation[TV_ParameterizedTruncatedNormal_dtype], stdevs: _atypes.TensorFuzzingAnnotation[TV_ParameterizedTruncatedNormal_dtype], minvals: _atypes.TensorFuzzingAnnotation[TV_ParameterizedTruncatedNormal_dtype], maxvals: _atypes.TensorFuzzingAnnotation[TV_ParameterizedTruncatedNormal_dtype], seed: int, seed2: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_ParameterizedTruncatedNormal_dtype]:
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    _attr_dtype, _inputs_dtype = _execute.args_to_matching_eager([means, stdevs, minvals, maxvals], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64])
    means, stdevs, minvals, maxvals = _inputs_dtype
    _attr_T, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64])
    _inputs_flat = [shape, means, stdevs, minvals, maxvals]
    _attrs = ('seed', seed, 'seed2', seed2, 'dtype', _attr_dtype, 'T', _attr_T)
    _result = _execute.execute(b'ParameterizedTruncatedNormal', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ParameterizedTruncatedNormal', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result