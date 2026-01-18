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
def experimental_thread_pool_handle(num_threads: int, display_name: str, max_intra_op_parallelism: int=1, container: str='', shared_name: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Resource]:
    """Creates a dataset that uses a custom thread pool to compute `input_dataset`.

  Args:
    num_threads: An `int`. The number of threads in the thread pool.
    display_name: A `string`.
      A human-readable name for the threads that may be visible in some
      visualizations.
      threadpool.
    max_intra_op_parallelism: An optional `int`. Defaults to `1`.
      The maximum degree of parallelism to use within operations that execute on this
      threadpool.
    container: An optional `string`. Defaults to `""`.
    shared_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `resource`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ExperimentalThreadPoolHandle', name, 'num_threads', num_threads, 'max_intra_op_parallelism', max_intra_op_parallelism, 'display_name', display_name, 'container', container, 'shared_name', shared_name)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return experimental_thread_pool_handle_eager_fallback(num_threads=num_threads, max_intra_op_parallelism=max_intra_op_parallelism, display_name=display_name, container=container, shared_name=shared_name, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
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
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ExperimentalThreadPoolHandle', num_threads=num_threads, display_name=display_name, max_intra_op_parallelism=max_intra_op_parallelism, container=container, shared_name=shared_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('num_threads', _op._get_attr_int('num_threads'), 'max_intra_op_parallelism', _op._get_attr_int('max_intra_op_parallelism'), 'display_name', _op.get_attr('display_name'), 'container', _op.get_attr('container'), 'shared_name', _op.get_attr('shared_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ExperimentalThreadPoolHandle', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result