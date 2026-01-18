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
def audio_spectrogram_eager_fallback(input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], window_size: int, stride: int, magnitude_squared: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    window_size = _execute.make_int(window_size, 'window_size')
    stride = _execute.make_int(stride, 'stride')
    if magnitude_squared is None:
        magnitude_squared = False
    magnitude_squared = _execute.make_bool(magnitude_squared, 'magnitude_squared')
    input = _ops.convert_to_tensor(input, _dtypes.float32)
    _inputs_flat = [input]
    _attrs = ('window_size', window_size, 'stride', stride, 'magnitude_squared', magnitude_squared)
    _result = _execute.execute(b'AudioSpectrogram', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('AudioSpectrogram', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result