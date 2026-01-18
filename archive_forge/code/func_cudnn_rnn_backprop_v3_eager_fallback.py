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
def cudnn_rnn_backprop_v3_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], input_h: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], input_c: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], params: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], sequence_lengths: _atypes.TensorFuzzingAnnotation[_atypes.Int32], output: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], output_h: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], output_c: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], output_backprop: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], output_h_backprop: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], output_c_backprop: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], reserve_space: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], host_reserved: _atypes.TensorFuzzingAnnotation[_atypes.Int8], rnn_mode: str, input_mode: str, direction: str, dropout: float, seed: int, seed2: int, num_proj: int, time_major: bool, name, ctx):
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
    if num_proj is None:
        num_proj = 0
    num_proj = _execute.make_int(num_proj, 'num_proj')
    if time_major is None:
        time_major = True
    time_major = _execute.make_bool(time_major, 'time_major')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([input, input_h, input_c, params, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64])
    input, input_h, input_c, params, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space = _inputs_T
    sequence_lengths = _ops.convert_to_tensor(sequence_lengths, _dtypes.int32)
    host_reserved = _ops.convert_to_tensor(host_reserved, _dtypes.int8)
    _inputs_flat = [input, input_h, input_c, params, sequence_lengths, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space, host_reserved]
    _attrs = ('T', _attr_T, 'rnn_mode', rnn_mode, 'input_mode', input_mode, 'direction', direction, 'dropout', dropout, 'seed', seed, 'seed2', seed2, 'num_proj', num_proj, 'time_major', time_major)
    _result = _execute.execute(b'CudnnRNNBackpropV3', 4, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CudnnRNNBackpropV3', _inputs_flat, _attrs, _result)
    _result = _CudnnRNNBackpropV3Output._make(_result)
    return _result