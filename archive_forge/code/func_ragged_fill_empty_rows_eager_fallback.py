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
def ragged_fill_empty_rows_eager_fallback(value_rowids: _atypes.TensorFuzzingAnnotation[_atypes.Int64], values: _atypes.TensorFuzzingAnnotation[TV_RaggedFillEmptyRows_T], nrows: _atypes.TensorFuzzingAnnotation[_atypes.Int64], default_value: _atypes.TensorFuzzingAnnotation[TV_RaggedFillEmptyRows_T], name, ctx):
    _attr_T, _inputs_T = _execute.args_to_matching_eager([values, default_value], ctx, [])
    values, default_value = _inputs_T
    value_rowids = _ops.convert_to_tensor(value_rowids, _dtypes.int64)
    nrows = _ops.convert_to_tensor(nrows, _dtypes.int64)
    _inputs_flat = [value_rowids, values, nrows, default_value]
    _attrs = ('T', _attr_T)
    _result = _execute.execute(b'RaggedFillEmptyRows', 4, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RaggedFillEmptyRows', _inputs_flat, _attrs, _result)
    _result = _RaggedFillEmptyRowsOutput._make(_result)
    return _result