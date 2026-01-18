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
def serialize_iterator(resource_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], external_state_policy: int=0, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Converts the given `resource_handle` representing an iterator to a variant tensor.

  Args:
    resource_handle: A `Tensor` of type `resource`.
      A handle to an iterator resource.
    external_state_policy: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SerializeIterator', name, resource_handle, 'external_state_policy', external_state_policy)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return serialize_iterator_eager_fallback(resource_handle, external_state_policy=external_state_policy, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if external_state_policy is None:
        external_state_policy = 0
    external_state_policy = _execute.make_int(external_state_policy, 'external_state_policy')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SerializeIterator', resource_handle=resource_handle, external_state_policy=external_state_policy, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('external_state_policy', _op._get_attr_int('external_state_policy'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SerializeIterator', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result