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
def tensor_list_element_shape(input_handle: _atypes.TensorFuzzingAnnotation[_atypes.Variant], shape_type: TV_TensorListElementShape_shape_type, name=None) -> _atypes.TensorFuzzingAnnotation[TV_TensorListElementShape_shape_type]:
    """The shape of the elements of the given list, as a tensor.

    input_handle: the list
    element_shape: the shape of elements of the list

  Args:
    input_handle: A `Tensor` of type `variant`.
    shape_type: A `tf.DType` from: `tf.int32, tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `shape_type`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorListElementShape', name, input_handle, 'shape_type', shape_type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_list_element_shape_eager_fallback(input_handle, shape_type=shape_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    shape_type = _execute.make_type(shape_type, 'shape_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorListElementShape', input_handle=input_handle, shape_type=shape_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('shape_type', _op._get_attr_type('shape_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorListElementShape', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result