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
def space_to_batch_nd_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_SpaceToBatchND_T], block_shape: _atypes.TensorFuzzingAnnotation[TV_SpaceToBatchND_Tblock_shape], paddings: _atypes.TensorFuzzingAnnotation[TV_SpaceToBatchND_Tpaddings], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_SpaceToBatchND_T]:
    _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [])
    _attr_Tblock_shape, (block_shape,) = _execute.args_to_matching_eager([block_shape], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _attr_Tpaddings, (paddings,) = _execute.args_to_matching_eager([paddings], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    _inputs_flat = [input, block_shape, paddings]
    _attrs = ('T', _attr_T, 'Tblock_shape', _attr_Tblock_shape, 'Tpaddings', _attr_Tpaddings)
    _result = _execute.execute(b'SpaceToBatchND', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SpaceToBatchND', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result