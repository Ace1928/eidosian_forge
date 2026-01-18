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
def tensor_list_push_back(input_handle: _atypes.TensorFuzzingAnnotation[_atypes.Variant], tensor: _atypes.TensorFuzzingAnnotation[TV_TensorListPushBack_element_dtype], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Returns a list which has the passed-in `Tensor` as last element and the other elements of the given list in `input_handle`.

  tensor: The tensor to put on the list.
  input_handle: The old list.
  output_handle: A list with the elements of the old list followed by tensor.
  element_dtype: the type of elements in the list.
  element_shape: a shape compatible with that of elements in the list.

  Args:
    input_handle: A `Tensor` of type `variant`.
    tensor: A `Tensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorListPushBack', name, input_handle, tensor)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_list_push_back_eager_fallback(input_handle, tensor, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorListPushBack', input_handle=input_handle, tensor=tensor, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('element_dtype', _op._get_attr_type('element_dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorListPushBack', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result