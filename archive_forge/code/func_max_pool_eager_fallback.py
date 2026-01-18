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
def max_pool_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_MaxPool_T], ksize, strides, padding: str, explicit_paddings, data_format: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_MaxPool_T]:
    if not isinstance(ksize, (list, tuple)):
        raise TypeError("Expected list for 'ksize' argument to 'max_pool' Op, not %r." % ksize)
    ksize = [_execute.make_int(_i, 'ksize') for _i in ksize]
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'max_pool' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if explicit_paddings is None:
        explicit_paddings = []
    if not isinstance(explicit_paddings, (list, tuple)):
        raise TypeError("Expected list for 'explicit_paddings' argument to 'max_pool' Op, not %r." % explicit_paddings)
    explicit_paddings = [_execute.make_int(_i, 'explicit_paddings') for _i in explicit_paddings]
    if data_format is None:
        data_format = 'NHWC'
    data_format = _execute.make_str(data_format, 'data_format')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.uint16, _dtypes.qint8], _dtypes.float32)
    _inputs_flat = [input]
    _attrs = ('T', _attr_T, 'ksize', ksize, 'strides', strides, 'padding', padding, 'explicit_paddings', explicit_paddings, 'data_format', data_format)
    _result = _execute.execute(b'MaxPool', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('MaxPool', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result