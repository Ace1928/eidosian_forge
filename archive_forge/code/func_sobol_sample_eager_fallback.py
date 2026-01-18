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
def sobol_sample_eager_fallback(dim: _atypes.TensorFuzzingAnnotation[_atypes.Int32], num_results: _atypes.TensorFuzzingAnnotation[_atypes.Int32], skip: _atypes.TensorFuzzingAnnotation[_atypes.Int32], dtype: TV_SobolSample_dtype, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_SobolSample_dtype]:
    if dtype is None:
        dtype = _dtypes.float32
    dtype = _execute.make_type(dtype, 'dtype')
    dim = _ops.convert_to_tensor(dim, _dtypes.int32)
    num_results = _ops.convert_to_tensor(num_results, _dtypes.int32)
    skip = _ops.convert_to_tensor(skip, _dtypes.int32)
    _inputs_flat = [dim, num_results, skip]
    _attrs = ('dtype', dtype)
    _result = _execute.execute(b'SobolSample', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SobolSample', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result