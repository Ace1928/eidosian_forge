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
def tensor_summary_eager_fallback(tensor: _atypes.TensorFuzzingAnnotation[TV_TensorSummary_T], description: str, labels, display_name: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    if description is None:
        description = ''
    description = _execute.make_str(description, 'description')
    if labels is None:
        labels = []
    if not isinstance(labels, (list, tuple)):
        raise TypeError("Expected list for 'labels' argument to 'tensor_summary' Op, not %r." % labels)
    labels = [_execute.make_str(_s, 'labels') for _s in labels]
    if display_name is None:
        display_name = ''
    display_name = _execute.make_str(display_name, 'display_name')
    _attr_T, (tensor,) = _execute.args_to_matching_eager([tensor], ctx, [])
    _inputs_flat = [tensor]
    _attrs = ('T', _attr_T, 'description', description, 'labels', labels, 'display_name', display_name)
    _result = _execute.execute(b'TensorSummary', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TensorSummary', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result