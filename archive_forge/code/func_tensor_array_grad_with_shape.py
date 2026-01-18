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
def tensor_array_grad_with_shape(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], flow_in: _atypes.TensorFuzzingAnnotation[_atypes.Float32], shape_to_prepend: _atypes.TensorFuzzingAnnotation[_atypes.Int32], source: str, name=None):
    """Creates a TensorArray for storing multiple gradients of values in the given handle.

  Similar to TensorArrayGradV3. However it creates an accumulator with an
  expanded shape compared to the input TensorArray whose gradient is being
  computed. This enables multiple gradients for the same TensorArray to be
  calculated using the same accumulator.

  Args:
    handle: A `Tensor` of type `resource`.
      The handle to the forward TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    shape_to_prepend: A `Tensor` of type `int32`.
      An int32 vector representing a shape. Elements in the gradient accumulator will
      have shape which is this shape_to_prepend value concatenated with shape of the
      elements in the TensorArray corresponding to the input handle.
    source: A `string`.
      The gradient source string, used to decide which gradient TensorArray
      to return.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (grad_handle, flow_out).

    grad_handle: A `Tensor` of type `resource`.
    flow_out: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorArrayGradWithShape', name, handle, flow_in, shape_to_prepend, 'source', source)
            _result = _TensorArrayGradWithShapeOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_array_grad_with_shape_eager_fallback(handle, flow_in, shape_to_prepend, source=source, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    source = _execute.make_str(source, 'source')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorArrayGradWithShape', handle=handle, flow_in=flow_in, shape_to_prepend=shape_to_prepend, source=source, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('source', _op.get_attr('source'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorArrayGradWithShape', _inputs_flat, _attrs, _result)
    _result = _TensorArrayGradWithShapeOutput._make(_result)
    return _result