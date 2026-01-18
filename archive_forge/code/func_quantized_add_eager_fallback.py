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
def quantized_add_eager_fallback(x: _atypes.TensorFuzzingAnnotation[TV_QuantizedAdd_T1], y: _atypes.TensorFuzzingAnnotation[TV_QuantizedAdd_T2], min_x: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_x: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_y: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_y: _atypes.TensorFuzzingAnnotation[_atypes.Float32], Toutput: TV_QuantizedAdd_Toutput, name, ctx):
    if Toutput is None:
        Toutput = _dtypes.qint32
    Toutput = _execute.make_type(Toutput, 'Toutput')
    _attr_T1, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16])
    _attr_T2, (y,) = _execute.args_to_matching_eager([y], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16])
    min_x = _ops.convert_to_tensor(min_x, _dtypes.float32)
    max_x = _ops.convert_to_tensor(max_x, _dtypes.float32)
    min_y = _ops.convert_to_tensor(min_y, _dtypes.float32)
    max_y = _ops.convert_to_tensor(max_y, _dtypes.float32)
    _inputs_flat = [x, y, min_x, max_x, min_y, max_y]
    _attrs = ('T1', _attr_T1, 'T2', _attr_T2, 'Toutput', Toutput)
    _result = _execute.execute(b'QuantizedAdd', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizedAdd', _inputs_flat, _attrs, _result)
    _result = _QuantizedAddOutput._make(_result)
    return _result