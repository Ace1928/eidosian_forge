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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('quantization.fake_quant_with_min_max_args_gradient', v1=['quantization.fake_quant_with_min_max_args_gradient', 'fake_quant_with_min_max_args_gradient'])
@deprecated_endpoints('fake_quant_with_min_max_args_gradient')
def fake_quant_with_min_max_args_gradient(gradients: _atypes.TensorFuzzingAnnotation[_atypes.Float32], inputs: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min: float=-6, max: float=6, num_bits: int=8, narrow_range: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Compute gradients for a FakeQuantWithMinMaxArgs operation.

  Args:
    gradients: A `Tensor` of type `float32`.
      Backpropagated gradients above the FakeQuantWithMinMaxArgs operation.
    inputs: A `Tensor` of type `float32`.
      Values passed as inputs to the FakeQuantWithMinMaxArgs operation.
    min: An optional `float`. Defaults to `-6`.
    max: An optional `float`. Defaults to `6`.
    num_bits: An optional `int`. Defaults to `8`.
    narrow_range: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'FakeQuantWithMinMaxArgsGradient', name, gradients, inputs, 'min', min, 'max', max, 'num_bits', num_bits, 'narrow_range', narrow_range)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_fake_quant_with_min_max_args_gradient((gradients, inputs, min, max, num_bits, narrow_range, name), None)
            if _result is not NotImplemented:
                return _result
            return fake_quant_with_min_max_args_gradient_eager_fallback(gradients, inputs, min=min, max=max, num_bits=num_bits, narrow_range=narrow_range, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(fake_quant_with_min_max_args_gradient, (), dict(gradients=gradients, inputs=inputs, min=min, max=max, num_bits=num_bits, narrow_range=narrow_range, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_fake_quant_with_min_max_args_gradient((gradients, inputs, min, max, num_bits, narrow_range, name), None)
        if _result is not NotImplemented:
            return _result
    if min is None:
        min = -6
    min = _execute.make_float(min, 'min')
    if max is None:
        max = 6
    max = _execute.make_float(max, 'max')
    if num_bits is None:
        num_bits = 8
    num_bits = _execute.make_int(num_bits, 'num_bits')
    if narrow_range is None:
        narrow_range = False
    narrow_range = _execute.make_bool(narrow_range, 'narrow_range')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('FakeQuantWithMinMaxArgsGradient', gradients=gradients, inputs=inputs, min=min, max=max, num_bits=num_bits, narrow_range=narrow_range, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(fake_quant_with_min_max_args_gradient, (), dict(gradients=gradients, inputs=inputs, min=min, max=max, num_bits=num_bits, narrow_range=narrow_range, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('min', _op.get_attr('min'), 'max', _op.get_attr('max'), 'num_bits', _op._get_attr_int('num_bits'), 'narrow_range', _op._get_attr_bool('narrow_range'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('FakeQuantWithMinMaxArgsGradient', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result