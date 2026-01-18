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
def depth_to_space_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_DepthToSpace_T], block_size: int, data_format: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_DepthToSpace_T]:
    block_size = _execute.make_int(block_size, 'block_size')
    if data_format is None:
        data_format = 'NHWC'
    data_format = _execute.make_str(data_format, 'data_format')
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
    _inputs_flat = [input]
    _attrs = ('T', _attr_T, 'block_size', block_size, 'data_format', data_format)
    _result = _execute.execute(b'DepthToSpace', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DepthToSpace', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result