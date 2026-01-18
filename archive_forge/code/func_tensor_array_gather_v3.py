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
def tensor_array_gather_v3(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], indices: _atypes.TensorFuzzingAnnotation[_atypes.Int32], flow_in: _atypes.TensorFuzzingAnnotation[_atypes.Float32], dtype: TV_TensorArrayGatherV3_dtype, element_shape=None, name=None) -> _atypes.TensorFuzzingAnnotation[TV_TensorArrayGatherV3_dtype]:
    """Gather specific elements from the TensorArray into output `value`.

  All elements selected by `indices` must have the same shape.

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    indices: A `Tensor` of type `int32`.
      The locations in the TensorArray from which to read tensor elements.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    dtype: A `tf.DType`. The type of the elem that is returned.
    element_shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
      The expected shape of an element, if known. Used to
      validate the shapes of TensorArray elements. If this shape is not
      fully specified, gathering zero-size TensorArrays is an error.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorArrayGatherV3', name, handle, indices, flow_in, 'dtype', dtype, 'element_shape', element_shape)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_array_gather_v3_eager_fallback(handle, indices, flow_in, dtype=dtype, element_shape=element_shape, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    dtype = _execute.make_type(dtype, 'dtype')
    if element_shape is None:
        element_shape = None
    element_shape = _execute.make_shape(element_shape, 'element_shape')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorArrayGatherV3', handle=handle, indices=indices, flow_in=flow_in, dtype=dtype, element_shape=element_shape, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'element_shape', _op.get_attr('element_shape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorArrayGatherV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result