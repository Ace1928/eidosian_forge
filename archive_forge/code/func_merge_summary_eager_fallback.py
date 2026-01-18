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
def merge_summary_eager_fallback(inputs: List[_atypes.TensorFuzzingAnnotation[_atypes.String]], name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("Expected list for 'inputs' argument to 'merge_summary' Op, not %r." % inputs)
    _attr_N = len(inputs)
    inputs = _ops.convert_n_to_tensor(inputs, _dtypes.string)
    _inputs_flat = list(inputs)
    _attrs = ('N', _attr_N)
    _result = _execute.execute(b'MergeSummary', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('MergeSummary', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result