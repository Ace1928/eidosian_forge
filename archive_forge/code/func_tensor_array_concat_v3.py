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
def tensor_array_concat_v3(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], flow_in: _atypes.TensorFuzzingAnnotation[_atypes.Float32], dtype: TV_TensorArrayConcatV3_dtype, element_shape_except0=None, name=None):
    """Concat the elements from the TensorArray into value `value`.

  Takes `T` elements of shapes

    ```
    (n0 x d0 x d1 x ...), (n1 x d0 x d1 x ...), ..., (n(T-1) x d0 x d1 x ...)
    ```

  and concatenates them into a Tensor of shape:

    ```
    (n0 + n1 + ... + n(T-1) x d0 x d1 x ...)
    ```

  All elements must have the same shape (excepting the first dimension).

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    dtype: A `tf.DType`. The type of the elem that is returned.
    element_shape_except0: An optional `tf.TensorShape` or list of `ints`. Defaults to `None`.
      The expected shape of an element, if known,
      excluding the first dimension. Used to validate the shapes of
      TensorArray elements. If this shape is not fully specified, concatenating
      zero-size TensorArrays is an error.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (value, lengths).

    value: A `Tensor` of type `dtype`.
    lengths: A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorArrayConcatV3', name, handle, flow_in, 'dtype', dtype, 'element_shape_except0', element_shape_except0)
            _result = _TensorArrayConcatV3Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_array_concat_v3_eager_fallback(handle, flow_in, dtype=dtype, element_shape_except0=element_shape_except0, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    dtype = _execute.make_type(dtype, 'dtype')
    if element_shape_except0 is None:
        element_shape_except0 = None
    element_shape_except0 = _execute.make_shape(element_shape_except0, 'element_shape_except0')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorArrayConcatV3', handle=handle, flow_in=flow_in, dtype=dtype, element_shape_except0=element_shape_except0, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'element_shape_except0', _op.get_attr('element_shape_except0'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorArrayConcatV3', _inputs_flat, _attrs, _result)
    _result = _TensorArrayConcatV3Output._make(_result)
    return _result