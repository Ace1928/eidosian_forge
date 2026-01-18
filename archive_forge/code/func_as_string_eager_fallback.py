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
def as_string_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_AsString_T], precision: int, scientific: bool, shortest: bool, width: int, fill: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    if precision is None:
        precision = -1
    precision = _execute.make_int(precision, 'precision')
    if scientific is None:
        scientific = False
    scientific = _execute.make_bool(scientific, 'scientific')
    if shortest is None:
        shortest = False
    shortest = _execute.make_bool(shortest, 'shortest')
    if width is None:
        width = -1
    width = _execute.make_int(width, 'width')
    if fill is None:
        fill = ''
    fill = _execute.make_str(fill, 'fill')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.complex64, _dtypes.complex128, _dtypes.bool, _dtypes.variant, _dtypes.string])
    _inputs_flat = [input]
    _attrs = ('T', _attr_T, 'precision', precision, 'scientific', scientific, 'shortest', shortest, 'width', width, 'fill', fill)
    _result = _execute.execute(b'AsString', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('AsString', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result