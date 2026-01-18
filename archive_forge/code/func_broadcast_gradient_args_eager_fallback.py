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
def broadcast_gradient_args_eager_fallback(s0: _atypes.TensorFuzzingAnnotation[TV_BroadcastGradientArgs_T], s1: _atypes.TensorFuzzingAnnotation[TV_BroadcastGradientArgs_T], name, ctx):
    _attr_T, _inputs_T = _execute.args_to_matching_eager([s0, s1], ctx, [_dtypes.int32, _dtypes.int64], _dtypes.int32)
    s0, s1 = _inputs_T
    _inputs_flat = [s0, s1]
    _attrs = ('T', _attr_T)
    _result = _execute.execute(b'BroadcastGradientArgs', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BroadcastGradientArgs', _inputs_flat, _attrs, _result)
    _result = _BroadcastGradientArgsOutput._make(_result)
    return _result