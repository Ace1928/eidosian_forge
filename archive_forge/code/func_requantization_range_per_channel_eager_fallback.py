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
def requantization_range_per_channel_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_RequantizationRangePerChannel_T], input_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], input_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], clip_value_max: float, name, ctx):
    clip_value_max = _execute.make_float(clip_value_max, 'clip_value_max')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.qint16, _dtypes.quint16], _dtypes.qint32)
    input_min = _ops.convert_to_tensor(input_min, _dtypes.float32)
    input_max = _ops.convert_to_tensor(input_max, _dtypes.float32)
    _inputs_flat = [input, input_min, input_max]
    _attrs = ('T', _attr_T, 'clip_value_max', clip_value_max)
    _result = _execute.execute(b'RequantizationRangePerChannel', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RequantizationRangePerChannel', _inputs_flat, _attrs, _result)
    _result = _RequantizationRangePerChannelOutput._make(_result)
    return _result