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
def decode_jpeg_eager_fallback(contents: _atypes.TensorFuzzingAnnotation[_atypes.String], channels: int, ratio: int, fancy_upscaling: bool, try_recover_truncated: bool, acceptable_fraction: float, dct_method: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.UInt8]:
    if channels is None:
        channels = 0
    channels = _execute.make_int(channels, 'channels')
    if ratio is None:
        ratio = 1
    ratio = _execute.make_int(ratio, 'ratio')
    if fancy_upscaling is None:
        fancy_upscaling = True
    fancy_upscaling = _execute.make_bool(fancy_upscaling, 'fancy_upscaling')
    if try_recover_truncated is None:
        try_recover_truncated = False
    try_recover_truncated = _execute.make_bool(try_recover_truncated, 'try_recover_truncated')
    if acceptable_fraction is None:
        acceptable_fraction = 1
    acceptable_fraction = _execute.make_float(acceptable_fraction, 'acceptable_fraction')
    if dct_method is None:
        dct_method = ''
    dct_method = _execute.make_str(dct_method, 'dct_method')
    contents = _ops.convert_to_tensor(contents, _dtypes.string)
    _inputs_flat = [contents]
    _attrs = ('channels', channels, 'ratio', ratio, 'fancy_upscaling', fancy_upscaling, 'try_recover_truncated', try_recover_truncated, 'acceptable_fraction', acceptable_fraction, 'dct_method', dct_method)
    _result = _execute.execute(b'DecodeJpeg', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DecodeJpeg', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result