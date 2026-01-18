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
def cudnn_rnn_backprop_v3(input: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], input_h: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], input_c: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], params: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], sequence_lengths: _atypes.TensorFuzzingAnnotation[_atypes.Int32], output: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], output_h: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], output_c: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], output_backprop: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], output_h_backprop: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], output_c_backprop: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], reserve_space: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNBackpropV3_T], host_reserved: _atypes.TensorFuzzingAnnotation[_atypes.Int8], rnn_mode: str='lstm', input_mode: str='linear_input', direction: str='unidirectional', dropout: float=0, seed: int=0, seed2: int=0, num_proj: int=0, time_major: bool=True, name=None):
    """Backprop step of CudnnRNNV3.

  Compute the backprop of both data and weights in a RNN. Takes an extra
      "sequence_lengths" input than CudnnRNNBackprop.

  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicates whether there is a linear projection between the input and
      the actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used. Should be
    "unidirectional" or "bidirectional".
  dropout: Dropout probability. When set to 0., dropout is disabled.
  seed: The 1st part of a seed to initialize dropout.
  seed2: The 2nd part of a seed to initialize dropout.
  input: If time_major is true, this is a 3-D tensor with the shape of
      [seq_length, batch_size, input_size]. If time_major is false, the shape is
      [batch_size, seq_length, input_size].
  input_h: If time_major is true, this is a 3-D tensor with the shape of
      [num_layer * dir, batch_size, num_units]. If time_major is false, the shape
      is [batch_size, num_layer * dir, num_units].
  input_c: For LSTM, a 3-D tensor with the shape of
      [num_layer * dir, batch, num_units]. For other models, it is ignored.
  params: A 1-D tensor that contains the weights and biases in an opaque layout.
      The size must be created through CudnnRNNParamsSize, and initialized
      separately. Note that they might not be compatible across different
      generations. So it is a good idea to save and restore
  sequence_lengths: a vector of lengths of each input sequence.
  output: If time_major is true, this is a 3-D tensor with the shape of
      [seq_length, batch_size, dir * num_units]. If time_major is false, the
      shape is [batch_size, seq_length, dir * num_units].
  output_h: The same shape has input_h.
  output_c: The same shape as input_c for LSTM. An empty tensor for other models.
  output_backprop: A 3-D tensor with the same shape as output in the forward pass.
  output_h_backprop: A 3-D tensor with the same shape as output_h in the forward
      pass.
  output_c_backprop: A 3-D tensor with the same shape as output_c in the forward
      pass.
  time_major: Indicates whether the input/output format is time major or batch
      major.
  reserve_space: The same reserve_space produced in the forward operation.
  input_backprop: The backprop to input in the forward pass. Has the same shape
      as input.
  input_h_backprop: The backprop to input_h in the forward pass. Has the same
      shape as input_h.
  input_c_backprop: The backprop to input_c in the forward pass. Has the same
      shape as input_c.
  params_backprop: The backprop to the params buffer in the forward pass. Has the
      same shape as params.

  Args:
    input: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    input_h: A `Tensor`. Must have the same type as `input`.
    input_c: A `Tensor`. Must have the same type as `input`.
    params: A `Tensor`. Must have the same type as `input`.
    sequence_lengths: A `Tensor` of type `int32`.
    output: A `Tensor`. Must have the same type as `input`.
    output_h: A `Tensor`. Must have the same type as `input`.
    output_c: A `Tensor`. Must have the same type as `input`.
    output_backprop: A `Tensor`. Must have the same type as `input`.
    output_h_backprop: A `Tensor`. Must have the same type as `input`.
    output_c_backprop: A `Tensor`. Must have the same type as `input`.
    reserve_space: A `Tensor`. Must have the same type as `input`.
    host_reserved: A `Tensor` of type `int8`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    num_proj: An optional `int`. Defaults to `0`.
    time_major: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (input_backprop, input_h_backprop, input_c_backprop, params_backprop).

    input_backprop: A `Tensor`. Has the same type as `input`.
    input_h_backprop: A `Tensor`. Has the same type as `input`.
    input_c_backprop: A `Tensor`. Has the same type as `input`.
    params_backprop: A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CudnnRNNBackpropV3', name, input, input_h, input_c, params, sequence_lengths, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space, host_reserved, 'rnn_mode', rnn_mode, 'input_mode', input_mode, 'direction', direction, 'dropout', dropout, 'seed', seed, 'seed2', seed2, 'num_proj', num_proj, 'time_major', time_major)
            _result = _CudnnRNNBackpropV3Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return cudnn_rnn_backprop_v3_eager_fallback(input, input_h, input_c, params, sequence_lengths, output, output_h, output_c, output_backprop, output_h_backprop, output_c_backprop, reserve_space, host_reserved, rnn_mode=rnn_mode, input_mode=input_mode, direction=direction, dropout=dropout, seed=seed, seed2=seed2, num_proj=num_proj, time_major=time_major, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
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
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CudnnRNNBackpropV3', input=input, input_h=input_h, input_c=input_c, params=params, sequence_lengths=sequence_lengths, output=output, output_h=output_h, output_c=output_c, output_backprop=output_backprop, output_h_backprop=output_h_backprop, output_c_backprop=output_c_backprop, reserve_space=reserve_space, host_reserved=host_reserved, rnn_mode=rnn_mode, input_mode=input_mode, direction=direction, dropout=dropout, seed=seed, seed2=seed2, num_proj=num_proj, time_major=time_major, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'rnn_mode', _op.get_attr('rnn_mode'), 'input_mode', _op.get_attr('input_mode'), 'direction', _op.get_attr('direction'), 'dropout', _op.get_attr('dropout'), 'seed', _op._get_attr_int('seed'), 'seed2', _op._get_attr_int('seed2'), 'num_proj', _op._get_attr_int('num_proj'), 'time_major', _op._get_attr_bool('time_major'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CudnnRNNBackpropV3', _inputs_flat, _attrs, _result)
    _result = _CudnnRNNBackpropV3Output._make(_result)
    return _result