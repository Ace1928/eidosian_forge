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
def anonymous_mutable_hash_table_of_tensors_eager_fallback(key_dtype: TV_AnonymousMutableHashTableOfTensors_key_dtype, value_dtype: TV_AnonymousMutableHashTableOfTensors_value_dtype, value_shape, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    key_dtype = _execute.make_type(key_dtype, 'key_dtype')
    value_dtype = _execute.make_type(value_dtype, 'value_dtype')
    if value_shape is None:
        value_shape = []
    value_shape = _execute.make_shape(value_shape, 'value_shape')
    _inputs_flat = []
    _attrs = ('key_dtype', key_dtype, 'value_dtype', value_dtype, 'value_shape', value_shape)
    _result = _execute.execute(b'AnonymousMutableHashTableOfTensors', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('AnonymousMutableHashTableOfTensors', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result