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
def quantized_reshape(tensor: _atypes.TensorFuzzingAnnotation[TV_QuantizedReshape_T], shape: _atypes.TensorFuzzingAnnotation[TV_QuantizedReshape_Tshape], input_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], input_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name=None):
    """Reshapes a quantized tensor as per the Reshape op.

  ```

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Defines the shape of the output tensor.
    input_min: A `Tensor` of type `float32`. The minimum value of the input.
    input_max: A `Tensor` of type `float32`. The maximum value of the input.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, output_min, output_max).

    output: A `Tensor`. Has the same type as `tensor`.
    output_min: A `Tensor` of type `float32`.
    output_max: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizedReshape', name, tensor, shape, input_min, input_max)
            _result = _QuantizedReshapeOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return quantized_reshape_eager_fallback(tensor, shape, input_min, input_max, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizedReshape', tensor=tensor, shape=shape, input_min=input_min, input_max=input_max, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tshape', _op._get_attr_type('Tshape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizedReshape', _inputs_flat, _attrs, _result)
    _result = _QuantizedReshapeOutput._make(_result)
    return _result