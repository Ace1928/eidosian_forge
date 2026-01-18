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
def scatter_nd_eager_fallback(indices: _atypes.TensorFuzzingAnnotation[TV_ScatterNd_Tindices], updates: _atypes.TensorFuzzingAnnotation[TV_ScatterNd_T], shape: _atypes.TensorFuzzingAnnotation[TV_ScatterNd_Tindices], name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_ScatterNd_T]:
    _attr_T, (updates,) = _execute.args_to_matching_eager([updates], ctx, [])
    _attr_Tindices, _inputs_Tindices = _execute.args_to_matching_eager([indices, shape], ctx, [_dtypes.int16, _dtypes.int32, _dtypes.int64])
    indices, shape = _inputs_Tindices
    _inputs_flat = [indices, updates, shape]
    _attrs = ('T', _attr_T, 'Tindices', _attr_Tindices)
    _result = _execute.execute(b'ScatterNd', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ScatterNd', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result