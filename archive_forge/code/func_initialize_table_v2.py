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
def initialize_table_v2(table_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], keys: _atypes.TensorFuzzingAnnotation[TV_InitializeTableV2_Tkey], values: _atypes.TensorFuzzingAnnotation[TV_InitializeTableV2_Tval], name=None):
    """Table initializer that takes two tensors for keys and values respectively.

  Args:
    table_handle: A `Tensor` of type `resource`.
      Handle to a table which will be initialized.
    keys: A `Tensor`. Keys of type Tkey.
    values: A `Tensor`. Values of type Tval.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'InitializeTableV2', name, table_handle, keys, values)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return initialize_table_v2_eager_fallback(table_handle, keys, values, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('InitializeTableV2', table_handle=table_handle, keys=keys, values=values, name=name)
    return _op