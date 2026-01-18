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
def reader_restore_state_v2(reader_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], state: _atypes.TensorFuzzingAnnotation[_atypes.String], name=None):
    """Restore a reader to a previously saved state.

  Not all Readers support being restored, so this can produce an
  Unimplemented error.

  Args:
    reader_handle: A `Tensor` of type `resource`. Handle to a Reader.
    state: A `Tensor` of type `string`.
      Result of a ReaderSerializeState of a Reader with type
      matching reader_handle.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ReaderRestoreStateV2', name, reader_handle, state)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return reader_restore_state_v2_eager_fallback(reader_handle, state, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ReaderRestoreStateV2', reader_handle=reader_handle, state=state, name=name)
    return _op