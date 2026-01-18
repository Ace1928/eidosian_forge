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
def experimental_thread_pool_handle_eager_fallback(num_threads: int, display_name: str, max_intra_op_parallelism: int, container: str, shared_name: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    num_threads = _execute.make_int(num_threads, 'num_threads')
    display_name = _execute.make_str(display_name, 'display_name')
    if max_intra_op_parallelism is None:
        max_intra_op_parallelism = 1
    max_intra_op_parallelism = _execute.make_int(max_intra_op_parallelism, 'max_intra_op_parallelism')
    if container is None:
        container = ''
    container = _execute.make_str(container, 'container')
    if shared_name is None:
        shared_name = ''
    shared_name = _execute.make_str(shared_name, 'shared_name')
    _inputs_flat = []
    _attrs = ('num_threads', num_threads, 'max_intra_op_parallelism', max_intra_op_parallelism, 'display_name', display_name, 'container', container, 'shared_name', shared_name)
    _result = _execute.execute(b'ExperimentalThreadPoolHandle', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ExperimentalThreadPoolHandle', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result