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
def lookup_table_insert_v2_eager_fallback(table_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], keys: _atypes.TensorFuzzingAnnotation[TV_LookupTableInsertV2_Tin], values: _atypes.TensorFuzzingAnnotation[TV_LookupTableInsertV2_Tout], name, ctx):
    _attr_Tin, (keys,) = _execute.args_to_matching_eager([keys], ctx, [])
    _attr_Tout, (values,) = _execute.args_to_matching_eager([values], ctx, [])
    table_handle = _ops.convert_to_tensor(table_handle, _dtypes.resource)
    _inputs_flat = [table_handle, keys, values]
    _attrs = ('Tin', _attr_Tin, 'Tout', _attr_Tout)
    _result = _execute.execute(b'LookupTableInsertV2', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result