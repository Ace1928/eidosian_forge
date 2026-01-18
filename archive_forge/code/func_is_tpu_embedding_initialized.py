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
def is_tpu_embedding_initialized(config: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Bool]:
    """Whether TPU Embedding is initialized in a distributed TPU system.

  Args:
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'IsTPUEmbeddingInitialized', name, 'config', config)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return is_tpu_embedding_initialized_eager_fallback(config=config, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if config is None:
        config = ''
    config = _execute.make_str(config, 'config')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('IsTPUEmbeddingInitialized', config=config, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('config', _op.get_attr('config'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('IsTPUEmbeddingInitialized', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result