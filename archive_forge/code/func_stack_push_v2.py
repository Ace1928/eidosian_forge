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
def stack_push_v2(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], elem: _atypes.TensorFuzzingAnnotation[TV_StackPushV2_T], swap_memory: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_StackPushV2_T]:
    """Push an element onto the stack.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a stack.
    elem: A `Tensor`. The tensor to be pushed onto the stack.
    swap_memory: An optional `bool`. Defaults to `False`.
      Swap `elem` to CPU. Default to false.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `elem`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StackPushV2', name, handle, elem, 'swap_memory', swap_memory)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return stack_push_v2_eager_fallback(handle, elem, swap_memory=swap_memory, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if swap_memory is None:
        swap_memory = False
    swap_memory = _execute.make_bool(swap_memory, 'swap_memory')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StackPushV2', handle=handle, elem=elem, swap_memory=swap_memory, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'swap_memory', _op._get_attr_bool('swap_memory'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StackPushV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result