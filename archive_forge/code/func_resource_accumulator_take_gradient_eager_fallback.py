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
def resource_accumulator_take_gradient_eager_fallback(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], num_required: _atypes.TensorFuzzingAnnotation[_atypes.Int32], dtype: TV_ResourceAccumulatorTakeGradient_dtype, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_ResourceAccumulatorTakeGradient_dtype]:
    dtype = _execute.make_type(dtype, 'dtype')
    handle = _ops.convert_to_tensor(handle, _dtypes.resource)
    num_required = _ops.convert_to_tensor(num_required, _dtypes.int32)
    _inputs_flat = [handle, num_required]
    _attrs = ('dtype', dtype)
    _result = _execute.execute(b'ResourceAccumulatorTakeGradient', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ResourceAccumulatorTakeGradient', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result