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
def take_many_sparse_from_tensors_map_eager_fallback(sparse_handles: _atypes.TensorFuzzingAnnotation[_atypes.Int64], dtype: TV_TakeManySparseFromTensorsMap_dtype, container: str, shared_name: str, name, ctx):
    dtype = _execute.make_type(dtype, 'dtype')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    sparse_handles = _ops.convert_to_tensor(sparse_handles, _dtypes.int64)
    _inputs_flat = [sparse_handles]
    _attrs = ('dtype', dtype, 'container', container, 'shared_name', shared_name)
    _result = _execute.execute(b'TakeManySparseFromTensorsMap', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TakeManySparseFromTensorsMap', _inputs_flat, _attrs, _result)
    _result = _TakeManySparseFromTensorsMapOutput._make(_result)
    return _result