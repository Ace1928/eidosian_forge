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
def in_top_k_eager_fallback(predictions: _atypes.TensorFuzzingAnnotation[_atypes.Float32], targets: _atypes.TensorFuzzingAnnotation[TV_InTopK_T], k: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Bool]:
    k = _execute.make_int(k, 'k')
    _attr_T, (targets,) = _execute.args_to_matching_eager([targets], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    predictions = _ops.convert_to_tensor(predictions, _dtypes.float32)
    _inputs_flat = [predictions, targets]
    _attrs = ('k', k, 'T', _attr_T)
    _result = _execute.execute(b'InTopK', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('InTopK', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result