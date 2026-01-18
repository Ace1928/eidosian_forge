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
def tensor_list_scatter_into_existing_list(input_handle: _atypes.TensorFuzzingAnnotation[_atypes.Variant], tensor: _atypes.TensorFuzzingAnnotation[TV_TensorListScatterIntoExistingList_element_dtype], indices: _atypes.TensorFuzzingAnnotation[_atypes.Int32], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Scatters tensor at indices in an input list.

  Each member of the TensorList corresponds to one row of the input tensor,
  specified by the given index (see `tf.gather`).

  input_handle: The list to scatter into.
  tensor: The input tensor.
  indices: The indices used to index into the list.
  output_handle: The TensorList.

  Args:
    input_handle: A `Tensor` of type `variant`.
    tensor: A `Tensor`.
    indices: A `Tensor` of type `int32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorListScatterIntoExistingList', name, input_handle, tensor, indices)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_list_scatter_into_existing_list_eager_fallback(input_handle, tensor, indices, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorListScatterIntoExistingList', input_handle=input_handle, tensor=tensor, indices=indices, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('element_dtype', _op._get_attr_type('element_dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorListScatterIntoExistingList', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result