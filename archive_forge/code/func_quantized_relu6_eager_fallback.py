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
def quantized_relu6_eager_fallback(features: _atypes.TensorFuzzingAnnotation[TV_QuantizedRelu6_Tinput], min_features: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_features: _atypes.TensorFuzzingAnnotation[_atypes.Float32], out_type: TV_QuantizedRelu6_out_type, name, ctx):
    if out_type is None:
        out_type = _dtypes.quint8
    out_type = _execute.make_type(out_type, 'out_type')
    _attr_Tinput, (features,) = _execute.args_to_matching_eager([features], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16])
    min_features = _ops.convert_to_tensor(min_features, _dtypes.float32)
    max_features = _ops.convert_to_tensor(max_features, _dtypes.float32)
    _inputs_flat = [features, min_features, max_features]
    _attrs = ('Tinput', _attr_Tinput, 'out_type', out_type)
    _result = _execute.execute(b'QuantizedRelu6', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizedRelu6', _inputs_flat, _attrs, _result)
    _result = _QuantizedRelu6Output._make(_result)
    return _result