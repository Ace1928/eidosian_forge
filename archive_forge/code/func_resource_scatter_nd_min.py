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
def resource_scatter_nd_min(ref: _atypes.TensorFuzzingAnnotation[_atypes.Resource], indices: _atypes.TensorFuzzingAnnotation[TV_ResourceScatterNdMin_Tindices], updates: _atypes.TensorFuzzingAnnotation[TV_ResourceScatterNdMin_T], use_locking: bool=True, name=None):
    """TODO: add doc.

  Args:
    ref: A `Tensor` of type `resource`.
      A resource handle. Must be from a VarHandleOp.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A Tensor. Must be one of the following types: int32, int64.
      A tensor of indices into ref.
    updates: A `Tensor`. A Tensor. Must have the same type as ref. A tensor of
      values whose element wise min is taken with ref.
    use_locking: An optional `bool`. Defaults to `True`.
      An optional bool. Defaults to True. If True, the assignment will
      be protected by a lock; otherwise the behavior is undefined,
      but may exhibit less contention.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResourceScatterNdMin', name, ref, indices, updates, 'use_locking', use_locking)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resource_scatter_nd_min_eager_fallback(ref, indices, updates, use_locking=use_locking, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if use_locking is None:
        use_locking = True
    use_locking = _execute.make_bool(use_locking, 'use_locking')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResourceScatterNdMin', ref=ref, indices=indices, updates=updates, use_locking=use_locking, name=name)
    return _op