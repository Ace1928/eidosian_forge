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
def batch_to_space_nd_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_BatchToSpaceND_T], block_shape: _atypes.TensorFuzzingAnnotation[TV_BatchToSpaceND_Tblock_shape], crops: _atypes.TensorFuzzingAnnotation[TV_BatchToSpaceND_Tcrops], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_BatchToSpaceND_T]:
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
    _attr_Tblock_shape, (block_shape,) = _execute.args_to_matching_eager([block_shape], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _attr_Tcrops, (crops,) = _execute.args_to_matching_eager([crops], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _inputs_flat = [input, block_shape, crops]
    _attrs = ('T', _attr_T, 'Tblock_shape', _attr_Tblock_shape, 'Tcrops', _attr_Tcrops)
    _result = _execute.execute(b'BatchToSpaceND', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BatchToSpaceND', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result