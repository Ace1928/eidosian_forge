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
def mfcc_eager_fallback(spectrogram: _atypes.TensorFuzzingAnnotation[_atypes.Float32], sample_rate: _atypes.TensorFuzzingAnnotation[_atypes.Int32], upper_frequency_limit: float, lower_frequency_limit: float, filterbank_channel_count: int, dct_coefficient_count: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    if upper_frequency_limit is None:
        upper_frequency_limit = 4000
    upper_frequency_limit = _execute.make_float(upper_frequency_limit, 'upper_frequency_limit')
    if lower_frequency_limit is None:
        lower_frequency_limit = 20
    lower_frequency_limit = _execute.make_float(lower_frequency_limit, 'lower_frequency_limit')
    if filterbank_channel_count is None:
        filterbank_channel_count = 40
    filterbank_channel_count = _execute.make_int(filterbank_channel_count, 'filterbank_channel_count')
    if dct_coefficient_count is None:
        dct_coefficient_count = 13
    dct_coefficient_count = _execute.make_int(dct_coefficient_count, 'dct_coefficient_count')
    spectrogram = _ops.convert_to_tensor(spectrogram, _dtypes.float32)
    sample_rate = _ops.convert_to_tensor(sample_rate, _dtypes.int32)
    _inputs_flat = [spectrogram, sample_rate]
    _attrs = ('upper_frequency_limit', upper_frequency_limit, 'lower_frequency_limit', lower_frequency_limit, 'filterbank_channel_count', filterbank_channel_count, 'dct_coefficient_count', dct_coefficient_count)
    _result = _execute.execute(b'Mfcc', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('Mfcc', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result