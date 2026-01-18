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
def tensor_array_concat(handle: _atypes.TensorFuzzingAnnotation[_atypes.String], flow_in: _atypes.TensorFuzzingAnnotation[_atypes.Float32], dtype: TV_TensorArrayConcat_dtype, element_shape_except0=None, name=None):
    """TODO: add doc.

  Args:
    handle: A `Tensor` of type mutable `string`.
    flow_in: A `Tensor` of type `float32`.
    dtype: A `tf.DType`.
    element_shape_except0: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (value, lengths).

    value: A `Tensor` of type `dtype`.
    lengths: A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("tensor_array_concat op does not support eager execution. Arg 'handle' is a ref.")
    dtype = _execute.make_type(dtype, 'dtype')
    if element_shape_except0 is None:
        element_shape_except0 = None
    element_shape_except0 = _execute.make_shape(element_shape_except0, 'element_shape_except0')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorArrayConcat', handle=handle, flow_in=flow_in, dtype=dtype, element_shape_except0=element_shape_except0, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'element_shape_except0', _op.get_attr('element_shape_except0'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorArrayConcat', _inputs_flat, _attrs, _result)
    _result = _TensorArrayConcatOutput._make(_result)
    return _result