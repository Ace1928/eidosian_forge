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
def tensor_list_set_item(input_handle: _atypes.TensorFuzzingAnnotation[_atypes.Variant], index: _atypes.TensorFuzzingAnnotation[_atypes.Int32], item: _atypes.TensorFuzzingAnnotation[TV_TensorListSetItem_element_dtype], resize_if_index_out_of_bounds: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Sets the index-th position of the list to contain the given tensor.

  input_handle: the list
  index: the position in the list to which the tensor will be assigned
  item: the element to be assigned to that position
  output_handle: the new list, with the element in the proper position

  Args:
    input_handle: A `Tensor` of type `variant`.
    index: A `Tensor` of type `int32`.
    item: A `Tensor`.
    resize_if_index_out_of_bounds: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorListSetItem', name, input_handle, index, item, 'resize_if_index_out_of_bounds', resize_if_index_out_of_bounds)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_list_set_item_eager_fallback(input_handle, index, item, resize_if_index_out_of_bounds=resize_if_index_out_of_bounds, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if resize_if_index_out_of_bounds is None:
        resize_if_index_out_of_bounds = False
    resize_if_index_out_of_bounds = _execute.make_bool(resize_if_index_out_of_bounds, 'resize_if_index_out_of_bounds')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorListSetItem', input_handle=input_handle, index=index, item=item, resize_if_index_out_of_bounds=resize_if_index_out_of_bounds, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('element_dtype', _op._get_attr_type('element_dtype'), 'resize_if_index_out_of_bounds', _op._get_attr_bool('resize_if_index_out_of_bounds'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorListSetItem', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result