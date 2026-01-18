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
def stack_v2_eager_fallback(max_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], elem_type: TV_StackV2_elem_type, stack_name: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    elem_type = _execute.make_type(elem_type, 'elem_type')
    if stack_name is None:
        stack_name = ''
    stack_name = _execute.make_str(stack_name, 'stack_name')
    max_size = _ops.convert_to_tensor(max_size, _dtypes.int32)
    _inputs_flat = [max_size]
    _attrs = ('elem_type', elem_type, 'stack_name', stack_name)
    _result = _execute.execute(b'StackV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('StackV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result