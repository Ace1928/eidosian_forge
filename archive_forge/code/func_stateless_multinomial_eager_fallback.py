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
def stateless_multinomial_eager_fallback(logits: _atypes.TensorFuzzingAnnotation[TV_StatelessMultinomial_T], num_samples: _atypes.TensorFuzzingAnnotation[_atypes.Int32], seed: _atypes.TensorFuzzingAnnotation[TV_StatelessMultinomial_Tseed], output_dtype: TV_StatelessMultinomial_output_dtype, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_StatelessMultinomial_output_dtype]:
    if output_dtype is None:
        output_dtype = _dtypes.int64
    output_dtype = _execute.make_type(output_dtype, 'output_dtype')
    _attr_T, (logits,) = _execute.args_to_matching_eager([logits], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    _attr_Tseed, (seed,) = _execute.args_to_matching_eager([seed], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int64)
    num_samples = _ops.convert_to_tensor(num_samples, _dtypes.int32)
    _inputs_flat = [logits, num_samples, seed]
    _attrs = ('T', _attr_T, 'Tseed', _attr_Tseed, 'output_dtype', output_dtype)
    _result = _execute.execute(b'StatelessMultinomial', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('StatelessMultinomial', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result