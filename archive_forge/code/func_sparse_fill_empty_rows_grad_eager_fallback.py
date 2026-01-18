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
def sparse_fill_empty_rows_grad_eager_fallback(reverse_index_map: _atypes.TensorFuzzingAnnotation[_atypes.Int64], grad_values: _atypes.TensorFuzzingAnnotation[TV_SparseFillEmptyRowsGrad_T], name, ctx):
    _attr_T, (grad_values,) = _execute.args_to_matching_eager([grad_values], ctx, [])
    reverse_index_map = _ops.convert_to_tensor(reverse_index_map, _dtypes.int64)
    _inputs_flat = [reverse_index_map, grad_values]
    _attrs = ('T', _attr_T)
    _result = _execute.execute(b'SparseFillEmptyRowsGrad', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SparseFillEmptyRowsGrad', _inputs_flat, _attrs, _result)
    _result = _SparseFillEmptyRowsGradOutput._make(_result)
    return _result