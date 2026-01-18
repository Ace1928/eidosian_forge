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
def list_diff_eager_fallback(x: _atypes.TensorFuzzingAnnotation[TV_ListDiff_T], y: _atypes.TensorFuzzingAnnotation[TV_ListDiff_T], out_idx: TV_ListDiff_out_idx, name, ctx):
    if out_idx is None:
        out_idx = _dtypes.int32
    out_idx = _execute.make_type(out_idx, 'out_idx')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], ctx, [])
    x, y = _inputs_T
    _inputs_flat = [x, y]
    _attrs = ('T', _attr_T, 'out_idx', out_idx)
    _result = _execute.execute(b'ListDiff', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ListDiff', _inputs_flat, _attrs, _result)
    _result = _ListDiffOutput._make(_result)
    return _result