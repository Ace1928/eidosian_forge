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
def fused_batch_norm_grad(y_backprop: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormGrad_T], x: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormGrad_T], scale: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormGrad_T], reserve_space_1: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormGrad_T], reserve_space_2: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormGrad_T], epsilon: float=0.0001, data_format: str='NHWC', is_training: bool=True, name=None):
    """Gradient for batch normalization.

  Note that the size of 4D Tensors are defined by either "NHWC" or "NCHW".
  The size of 1D Tensors matches the dimension C of the 4D Tensors.

  Args:
    y_backprop: A `Tensor`. Must be one of the following types: `float32`.
      A 4D Tensor for the gradient with respect to y.
    x: A `Tensor`. Must have the same type as `y_backprop`.
      A 4D Tensor for input data.
    scale: A `Tensor`. Must have the same type as `y_backprop`.
      A 1D Tensor for scaling factor, to scale the normalized x.
    reserve_space_1: A `Tensor`. Must have the same type as `y_backprop`.
      When is_training is True, a 1D Tensor for the computed batch
      mean to be reused in gradient computation. When is_training is
      False, a 1D Tensor for the population mean to be reused in both
      1st and 2nd order gradient computation.
    reserve_space_2: A `Tensor`. Must have the same type as `y_backprop`.
      When is_training is True, a 1D Tensor for the computed batch
      variance (inverted variance in the cuDNN case) to be reused in
      gradient computation. When is_training is False, a 1D Tensor
      for the population variance to be reused in both 1st and 2nd
      order gradient computation.
    epsilon: An optional `float`. Defaults to `0.0001`.
      A small float number added to the variance of x.
    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
      The data format for y_backprop, x, x_backprop.
      Either "NHWC" (default) or "NCHW".
    is_training: An optional `bool`. Defaults to `True`.
      A bool value to indicate the operation is for training (default)
      or inference.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_backprop, scale_backprop, offset_backprop, reserve_space_3, reserve_space_4).

    x_backprop: A `Tensor`. Has the same type as `y_backprop`.
    scale_backprop: A `Tensor`. Has the same type as `y_backprop`.
    offset_backprop: A `Tensor`. Has the same type as `y_backprop`.
    reserve_space_3: A `Tensor`. Has the same type as `y_backprop`.
    reserve_space_4: A `Tensor`. Has the same type as `y_backprop`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'FusedBatchNormGrad', name, y_backprop, x, scale, reserve_space_1, reserve_space_2, 'epsilon', epsilon, 'data_format', data_format, 'is_training', is_training)
            _result = _FusedBatchNormGradOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return fused_batch_norm_grad_eager_fallback(y_backprop, x, scale, reserve_space_1, reserve_space_2, epsilon=epsilon, data_format=data_format, is_training=is_training, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if epsilon is None:
        epsilon = 0.0001
    epsilon = _execute.make_float(epsilon, 'epsilon')
    if data_format is None:
        data_format = 'NHWC'
    data_format = _execute.make_str(data_format, 'data_format')
    if is_training is None:
        is_training = True
    is_training = _execute.make_bool(is_training, 'is_training')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('FusedBatchNormGrad', y_backprop=y_backprop, x=x, scale=scale, reserve_space_1=reserve_space_1, reserve_space_2=reserve_space_2, epsilon=epsilon, data_format=data_format, is_training=is_training, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'epsilon', _op.get_attr('epsilon'), 'data_format', _op.get_attr('data_format'), 'is_training', _op._get_attr_bool('is_training'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('FusedBatchNormGrad', _inputs_flat, _attrs, _result)
    _result = _FusedBatchNormGradOutput._make(_result)
    return _result