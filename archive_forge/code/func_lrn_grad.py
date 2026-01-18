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
def lrn_grad(input_grads: _atypes.TensorFuzzingAnnotation[TV_LRNGrad_T], input_image: _atypes.TensorFuzzingAnnotation[TV_LRNGrad_T], output_image: _atypes.TensorFuzzingAnnotation[TV_LRNGrad_T], depth_radius: int=5, bias: float=1, alpha: float=1, beta: float=0.5, name=None) -> _atypes.TensorFuzzingAnnotation[TV_LRNGrad_T]:
    """Gradients for Local Response Normalization.

  Args:
    input_grads: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      4-D with shape `[batch, height, width, channels]`.
    input_image: A `Tensor`. Must have the same type as `input_grads`.
      4-D with shape `[batch, height, width, channels]`.
    output_image: A `Tensor`. Must have the same type as `input_grads`.
      4-D with shape `[batch, height, width, channels]`.
    depth_radius: An optional `int`. Defaults to `5`. A depth radius.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually > 0 to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input_grads`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'LRNGrad', name, input_grads, input_image, output_image, 'depth_radius', depth_radius, 'bias', bias, 'alpha', alpha, 'beta', beta)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return lrn_grad_eager_fallback(input_grads, input_image, output_image, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if depth_radius is None:
        depth_radius = 5
    depth_radius = _execute.make_int(depth_radius, 'depth_radius')
    if bias is None:
        bias = 1
    bias = _execute.make_float(bias, 'bias')
    if alpha is None:
        alpha = 1
    alpha = _execute.make_float(alpha, 'alpha')
    if beta is None:
        beta = 0.5
    beta = _execute.make_float(beta, 'beta')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('LRNGrad', input_grads=input_grads, input_image=input_image, output_image=output_image, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('depth_radius', _op._get_attr_int('depth_radius'), 'bias', _op.get_attr('bias'), 'alpha', _op.get_attr('alpha'), 'beta', _op.get_attr('beta'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('LRNGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result