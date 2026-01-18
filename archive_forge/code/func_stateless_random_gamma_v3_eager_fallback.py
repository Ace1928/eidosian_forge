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
def stateless_random_gamma_v3_eager_fallback(shape: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomGammaV3_shape_dtype], key: _atypes.TensorFuzzingAnnotation[_atypes.UInt64], counter: _atypes.TensorFuzzingAnnotation[_atypes.UInt64], alg: _atypes.TensorFuzzingAnnotation[_atypes.Int32], alpha: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomGammaV3_dtype], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_StatelessRandomGammaV3_dtype]:
    _attr_dtype, (alpha,) = _execute.args_to_matching_eager([alpha], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64])
    _attr_shape_dtype, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    key = _ops.convert_to_tensor(key, _dtypes.uint64)
    counter = _ops.convert_to_tensor(counter, _dtypes.uint64)
    alg = _ops.convert_to_tensor(alg, _dtypes.int32)
    _inputs_flat = [shape, key, counter, alg, alpha]
    _attrs = ('dtype', _attr_dtype, 'shape_dtype', _attr_shape_dtype)
    _result = _execute.execute(b'StatelessRandomGammaV3', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('StatelessRandomGammaV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result