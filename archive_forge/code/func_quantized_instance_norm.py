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
def quantized_instance_norm(x: _atypes.TensorFuzzingAnnotation[TV_QuantizedInstanceNorm_T], x_min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], x_max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], output_range_given: bool=False, given_y_min: float=0, given_y_max: float=0, variance_epsilon: float=1e-05, min_separation: float=0.001, name=None):
    """Quantized Instance normalization.

  Args:
    x: A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
      A 4D input Tensor.
    x_min: A `Tensor` of type `float32`.
      The value represented by the lowest quantized input.
    x_max: A `Tensor` of type `float32`.
      The value represented by the highest quantized input.
    output_range_given: An optional `bool`. Defaults to `False`.
      If True, `given_y_min` and `given_y_min`
      and `given_y_max` are used as the output range. Otherwise,
      the implementation computes the output range.
    given_y_min: An optional `float`. Defaults to `0`.
      Output in `y_min` if `output_range_given` is True.
    given_y_max: An optional `float`. Defaults to `0`.
      Output in `y_max` if `output_range_given` is True.
    variance_epsilon: An optional `float`. Defaults to `1e-05`.
      A small float number to avoid dividing by 0.
    min_separation: An optional `float`. Defaults to `0.001`.
      Minimum value of `y_max - y_min`
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (y, y_min, y_max).

    y: A `Tensor`. Has the same type as `x`.
    y_min: A `Tensor` of type `float32`.
    y_max: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'QuantizedInstanceNorm', name, x, x_min, x_max, 'output_range_given', output_range_given, 'given_y_min', given_y_min, 'given_y_max', given_y_max, 'variance_epsilon', variance_epsilon, 'min_separation', min_separation)
            _result = _QuantizedInstanceNormOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return quantized_instance_norm_eager_fallback(x, x_min, x_max, output_range_given=output_range_given, given_y_min=given_y_min, given_y_max=given_y_max, variance_epsilon=variance_epsilon, min_separation=min_separation, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if output_range_given is None:
        output_range_given = False
    output_range_given = _execute.make_bool(output_range_given, 'output_range_given')
    if given_y_min is None:
        given_y_min = 0
    given_y_min = _execute.make_float(given_y_min, 'given_y_min')
    if given_y_max is None:
        given_y_max = 0
    given_y_max = _execute.make_float(given_y_max, 'given_y_max')
    if variance_epsilon is None:
        variance_epsilon = 1e-05
    variance_epsilon = _execute.make_float(variance_epsilon, 'variance_epsilon')
    if min_separation is None:
        min_separation = 0.001
    min_separation = _execute.make_float(min_separation, 'min_separation')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('QuantizedInstanceNorm', x=x, x_min=x_min, x_max=x_max, output_range_given=output_range_given, given_y_min=given_y_min, given_y_max=given_y_max, variance_epsilon=variance_epsilon, min_separation=min_separation, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'output_range_given', _op._get_attr_bool('output_range_given'), 'given_y_min', _op.get_attr('given_y_min'), 'given_y_max', _op.get_attr('given_y_max'), 'variance_epsilon', _op.get_attr('variance_epsilon'), 'min_separation', _op.get_attr('min_separation'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('QuantizedInstanceNorm', _inputs_flat, _attrs, _result)
    _result = _QuantizedInstanceNormOutput._make(_result)
    return _result