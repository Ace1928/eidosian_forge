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
def irfft_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_IRFFT_Tcomplex], fft_length: _atypes.TensorFuzzingAnnotation[_atypes.Int32], Treal: TV_IRFFT_Treal, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_IRFFT_Treal]:
    if Treal is None:
        Treal = _dtypes.float32
    Treal = _execute.make_type(Treal, 'Treal')
    _attr_Tcomplex, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.complex64, _dtypes.complex128], _dtypes.complex64)
    fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
    _inputs_flat = [input, fft_length]
    _attrs = ('Treal', Treal, 'Tcomplex', _attr_Tcomplex)
    _result = _execute.execute(b'IRFFT', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('IRFFT', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result