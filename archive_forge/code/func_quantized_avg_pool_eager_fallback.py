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
def quantized_avg_pool_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_QuantizedAvgPool_T], min_input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], ksize, strides, padding: str, name, ctx):
    if not isinstance(ksize, (list, tuple)):
        raise TypeError("Expected list for 'ksize' argument to 'quantized_avg_pool' Op, not %r." % ksize)
    ksize = [_execute.make_int(_i, 'ksize') for _i in ksize]
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'quantized_avg_pool' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16])
    min_input = _ops.convert_to_tensor(min_input, _dtypes.float32)
    max_input = _ops.convert_to_tensor(max_input, _dtypes.float32)
    _inputs_flat = [input, min_input, max_input]
    _attrs = ('T', _attr_T, 'ksize', ksize, 'strides', strides, 'padding', padding)
    _result = _execute.execute(b'QuantizedAvgPool', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizedAvgPool', _inputs_flat, _attrs, _result)
    _result = _QuantizedAvgPoolOutput._make(_result)
    return _result