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
def tensor_array(size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], dtype: TV_TensorArray_dtype, dynamic_size: bool=False, clear_after_read: bool=True, tensor_array_name: str='', element_shape=None, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """TODO: add doc.

  Args:
    size: A `Tensor` of type `int32`.
    dtype: A `tf.DType`.
    dynamic_size: An optional `bool`. Defaults to `False`.
    clear_after_read: An optional `bool`. Defaults to `True`.
    tensor_array_name: An optional `string`. Defaults to `""`.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type mutable `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("tensor_array op does not support eager execution. Arg 'handle' is a ref.")
    dtype = _execute.make_type(dtype, 'dtype')
    if dynamic_size is None:
        dynamic_size = False
    dynamic_size = _execute.make_bool(dynamic_size, 'dynamic_size')
    if clear_after_read is None:
        clear_after_read = True
    clear_after_read = _execute.make_bool(clear_after_read, 'clear_after_read')
    if tensor_array_name is None:
        tensor_array_name = ''
    tensor_array_name = _execute.make_str(tensor_array_name, 'tensor_array_name')
    if element_shape is None:
        element_shape = None
    element_shape = _execute.make_shape(element_shape, 'element_shape')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorArray', size=size, dtype=dtype, dynamic_size=dynamic_size, clear_after_read=clear_after_read, tensor_array_name=tensor_array_name, element_shape=element_shape, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'dynamic_size', _op._get_attr_bool('dynamic_size'), 'clear_after_read', _op._get_attr_bool('clear_after_read'), 'tensor_array_name', _op.get_attr('tensor_array_name'), 'element_shape', _op.get_attr('element_shape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorArray', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result