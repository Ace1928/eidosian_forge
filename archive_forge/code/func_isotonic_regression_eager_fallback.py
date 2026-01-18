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
def isotonic_regression_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_IsotonicRegression_T], output_dtype: TV_IsotonicRegression_output_dtype, name, ctx):
    if output_dtype is None:
        output_dtype = _dtypes.float32
    output_dtype = _execute.make_type(output_dtype, 'output_dtype')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    _inputs_flat = [input]
    _attrs = ('T', _attr_T, 'output_dtype', output_dtype)
    _result = _execute.execute(b'IsotonicRegression', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('IsotonicRegression', _inputs_flat, _attrs, _result)
    _result = _IsotonicRegressionOutput._make(_result)
    return _result