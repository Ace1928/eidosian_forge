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
def select_eager_fallback(condition: _atypes.TensorFuzzingAnnotation[_atypes.Bool], x: _atypes.TensorFuzzingAnnotation[TV_Select_T], y: _atypes.TensorFuzzingAnnotation[TV_Select_T], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_Select_T]:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [])
    x, y = _inputs_T
    condition = _ops.convert_to_tensor(condition, _dtypes.bool)
    _inputs_flat = [condition, x, y]
    _attrs = ('T', _attr_T)
    _result = _execute.execute(b'Select', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('Select', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result