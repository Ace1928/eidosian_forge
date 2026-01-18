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
def quantize_v2_eager_fallback(input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_range: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_range: _atypes.TensorFuzzingAnnotation[_atypes.Float32], T: TV_QuantizeV2_T, mode: str, round_mode: str, narrow_range: bool, axis: int, ensure_minimum_range: float, name, ctx):
    T = _execute.make_type(T, 'T')
    if mode is None:
        mode = 'MIN_COMBINED'
    mode = _execute.make_str(mode, 'mode')
    if round_mode is None:
        round_mode = 'HALF_AWAY_FROM_ZERO'
    round_mode = _execute.make_str(round_mode, 'round_mode')
    if narrow_range is None:
        narrow_range = False
    narrow_range = _execute.make_bool(narrow_range, 'narrow_range')
    if axis is None:
        axis = -1
    axis = _execute.make_int(axis, 'axis')
    if ensure_minimum_range is None:
        ensure_minimum_range = 0.01
    ensure_minimum_range = _execute.make_float(ensure_minimum_range, 'ensure_minimum_range')
    input = _ops.convert_to_tensor(input, _dtypes.float32)
    min_range = _ops.convert_to_tensor(min_range, _dtypes.float32)
    max_range = _ops.convert_to_tensor(max_range, _dtypes.float32)
    _inputs_flat = [input, min_range, max_range]
    _attrs = ('T', T, 'mode', mode, 'round_mode', round_mode, 'narrow_range', narrow_range, 'axis', axis, 'ensure_minimum_range', ensure_minimum_range)
    _result = _execute.execute(b'QuantizeV2', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('QuantizeV2', _inputs_flat, _attrs, _result)
    _result = _QuantizeV2Output._make(_result)
    return _result