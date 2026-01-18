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
def cudnn_rnn_backprop_v2_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV2_T], input_h: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV2_T], input_c: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV2_T], params: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV2_T], output: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV2_T], output_h: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV2_T], output_c: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV2_T], output_backprop: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV2_T], output_h_backprop: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV2_T], output_c_backprop: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV2_T], reserve_space: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV2_T], host_reserved: _atypes.TensorFuzzingAnnotation[_atypes.Int8], rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, name, ctx):
    if rnn_mode is None:
        rnn_mode = 'lstm'
    rnn_mode = _execute.make_str(rnn_mode, 'rnn_mode')
    if input_mode is None:
        input_mode = 'linear_input'
    input_mode = _execute.make_str(input_mode, 'input_mode')
    if direction is None:
        direction = 'unidirectional'
    direction = _execute.make_str(direction, 'direction')
    if dropout is None:
        dropout = 0
    dropout = _execute.make_float(dropout, 'dropout')
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([input, input_h, input_c, params, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64])
    input, input_h, input_c, params, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space = _inputs_T
    host_reserved = _ops.convert_to_tensor(host_reserved, _dtypes.int8)
    _inputs_flat = [input, input_h, input_c, params, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space, host_reserved]
    _attrs = ('T', _attr_T, 'rnn_mode', rnn_mode, 'input_mode', input_mode, 'direction', direction, 'dropout', dropout, 'seed', seed, 'seed2', seed2)
    _result = _execute.execute(b'CudnnRNNBackpropV2', 4, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CudnnRNNBackpropV2', _inputs_flat, _attrs, _result)
    _result = _CudnnRNNBackpropV2Output._make(_result)
    return _result