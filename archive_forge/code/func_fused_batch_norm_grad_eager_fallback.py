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
def fused_batch_norm_grad_eager_fallback(y_backprop: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormGrad_T], x: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormGrad_T], scale: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormGrad_T], reserve_space_1: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormGrad_T], reserve_space_2: _atypes.TensorFuzzingAnnotation[TV_FusedBatchNormGrad_T], epsilon: float, data_format: str, is_training: bool, name, ctx):
    if epsilon is None:
        epsilon = 0.0001
    epsilon = _execute.make_float(epsilon, 'epsilon')
    if data_format is None:
        data_format = 'NHWC'
    data_format = _execute.make_str(data_format, 'data_format')
    if is_training is None:
        is_training = True
    is_training = _execute.make_bool(is_training, 'is_training')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([y_backprop, x, scale, reserve_space_1, reserve_space_2], ctx, [_dtypes.float32])
    y_backprop, x, scale, reserve_space_1, reserve_space_2 = _inputs_T
    _inputs_flat = [y_backprop, x, scale, reserve_space_1, reserve_space_2]
    _attrs = ('T', _attr_T, 'epsilon', epsilon, 'data_format', data_format, 'is_training', is_training)
    _result = _execute.execute(b'FusedBatchNormGrad', 5, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('FusedBatchNormGrad', _inputs_flat, _attrs, _result)
    _result = _FusedBatchNormGradOutput._make(_result)
    return _result