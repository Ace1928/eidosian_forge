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
def data_format_dim_map_eager_fallback(x: _atypes.TensorFuzzingAnnotation[TV_DataFormatDimMap_T], src_format: str, dst_format: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_DataFormatDimMap_T]:
    if src_format is None:
        src_format = 'NHWC'
    src_format = _execute.make_str(src_format, 'src_format')
    if dst_format is None:
        dst_format = 'NCHW'
    dst_format = _execute.make_str(dst_format, 'dst_format')
    _attr_T, (x,) = _execute.args_to_matching_eager([x], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _inputs_flat = [x]
    _attrs = ('T', _attr_T, 'src_format', src_format, 'dst_format', dst_format)
    _result = _execute.execute(b'DataFormatDimMap', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DataFormatDimMap', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result