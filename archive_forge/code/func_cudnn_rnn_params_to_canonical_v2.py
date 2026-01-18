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
def cudnn_rnn_params_to_canonical_v2(num_layers: _atypes.TensorFuzzingAnnotation[_atypes.Int32], num_units: _atypes.TensorFuzzingAnnotation[_atypes.Int32], input_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], params: _atypes.TensorFuzzingAnnotation[TV_CudnnRNNParamsToCanonicalV2_T], num_params_weights: int, num_params_biases: int, rnn_mode: str='lstm', input_mode: str='linear_input', direction: str='unidirectional', dropout: float=0, seed: int=0, seed2: int=0, num_proj: int=0, name=None):
    """Retrieves CudnnRNN params in canonical form. It supports the projection in LSTM.

  Retrieves a set of weights from the opaque params buffer that can be saved and
  restored in a way compatible with future runs.

  Note that the params buffer may not be compatible across different GPUs. So any
  save and restoration should be converted to and from the canonical weights and
  biases.

  num_layers: Specifies the number of layers in the RNN model.
  num_units: Specifies the size of the hidden state.
  input_size: Specifies the size of the input state.
  num_params_weights: number of weight parameter matrix for all layers.
  num_params_biases: number of bias parameter vector for all layers.
  weights: the canonical form of weights that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
  biases: the canonical form of biases that can be used for saving
      and restoration. They are more likely to be compatible across different
      generations.
  rnn_mode: Indicates the type of the RNN model.
  input_mode: Indicate whether there is a linear projection between the input and
      The actual computation before the first layer. 'skip_input' is only allowed
      when input_size == num_units; 'auto_select' implies 'skip_input' when
      input_size == num_units; otherwise, it implies 'linear_input'.
  direction: Indicates whether a bidirectional model will be used.
      dir = (direction == bidirectional) ? 2 : 1
  dropout: dropout probability. When set to 0., dropout is disabled.
  seed: the 1st part of a seed to initialize dropout.
  seed2: the 2nd part of a seed to initialize dropout.
  num_proj: The output dimensionality for the projection matrices. If None or 0,
      no projection is performed.

  Args:
    num_layers: A `Tensor` of type `int32`.
    num_units: A `Tensor` of type `int32`.
    input_size: A `Tensor` of type `int32`.
    params: A `Tensor`. Must be one of the following types: `bfloat16`, `half`, `float32`, `float64`.
    num_params_weights: An `int` that is `>= 1`.
    num_params_biases: An `int` that is `>= 1`.
    rnn_mode: An optional `string` from: `"rnn_relu", "rnn_tanh", "lstm", "gru"`. Defaults to `"lstm"`.
    input_mode: An optional `string` from: `"linear_input", "skip_input", "auto_select"`. Defaults to `"linear_input"`.
    direction: An optional `string` from: `"unidirectional", "bidirectional"`. Defaults to `"unidirectional"`.
    dropout: An optional `float`. Defaults to `0`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    num_proj: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (weights, biases).

    weights: A list of `num_params_weights` `Tensor` objects with the same type as `params`.
    biases: A list of `num_params_biases` `Tensor` objects with the same type as `params`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'CudnnRNNParamsToCanonicalV2', name, num_layers, num_units, input_size, params, 'num_params_weights', num_params_weights, 'num_params_biases', num_params_biases, 'rnn_mode', rnn_mode, 'input_mode', input_mode, 'direction', direction, 'dropout', dropout, 'seed', seed, 'seed2', seed2, 'num_proj', num_proj)
            _result = _CudnnRNNParamsToCanonicalV2Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return cudnn_rnn_params_to_canonical_v2_eager_fallback(num_layers, num_units, input_size, params, num_params_weights=num_params_weights, num_params_biases=num_params_biases, rnn_mode=rnn_mode, input_mode=input_mode, direction=direction, dropout=dropout, seed=seed, seed2=seed2, num_proj=num_proj, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    num_params_weights = _execute.make_int(num_params_weights, 'num_params_weights')
    num_params_biases = _execute.make_int(num_params_biases, 'num_params_biases')
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
    _, _, _op, _outputs = _op_def_library._apply_op_helper('CudnnRNNParamsToCanonicalV2', num_layers=num_layers, num_units=num_units, input_size=input_size, params=params, num_params_weights=num_params_weights, num_params_biases=num_params_biases, rnn_mode=rnn_mode, input_mode=input_mode, direction=direction, dropout=dropout, seed=seed, seed2=seed2, num_proj=num_proj, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'num_params_weights', _op._get_attr_int('num_params_weights'), 'num_params_biases', _op._get_attr_int('num_params_biases'), 'rnn_mode', _op.get_attr('rnn_mode'), 'input_mode', _op.get_attr('input_mode'), 'direction', _op.get_attr('direction'), 'dropout', _op.get_attr('dropout'), 'seed', _op._get_attr_int('seed'), 'seed2', _op._get_attr_int('seed2'), 'num_proj', _op._get_attr_int('num_proj'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('CudnnRNNParamsToCanonicalV2', _inputs_flat, _attrs, _result)
    _result = [_result[:num_params_weights]] + _result[num_params_weights:]
    _result = _result[:1] + [_result[1:]]
    _result = _CudnnRNNParamsToCanonicalV2Output._make(_result)
    return _result