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
def initialize_table(table_handle: _atypes.TensorFuzzingAnnotation[_atypes.String], keys: _atypes.TensorFuzzingAnnotation[TV_InitializeTable_Tkey], values: _atypes.TensorFuzzingAnnotation[TV_InitializeTable_Tval], name=None):
    """Table initializer that takes two tensors for keys and values respectively.

  Args:
    table_handle: A `Tensor` of type mutable `string`.
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
        raise RuntimeError("initialize_table op does not support eager execution. Arg 'table_handle' is a ref.")
    _, _, _op, _outputs = _op_def_library._apply_op_helper('InitializeTable', table_handle=table_handle, keys=keys, values=values, name=name)
    return _op