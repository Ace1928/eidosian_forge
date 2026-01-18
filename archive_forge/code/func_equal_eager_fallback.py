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
def equal_eager_fallback(x: _atypes.TensorFuzzingAnnotation[TV_Equal_T], y: _atypes.TensorFuzzingAnnotation[TV_Equal_T], incompatible_shape_error: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Bool]:
    if incompatible_shape_error is None:
        incompatible_shape_error = True
    incompatible_shape_error = _execute.make_bool(incompatible_shape_error, 'incompatible_shape_error')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [])
    x, y = _inputs_T
    _inputs_flat = [x, y]
    _attrs = ('T', _attr_T, 'incompatible_shape_error', incompatible_shape_error)
    _result = _execute.execute(b'Equal', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('Equal', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result