from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated(None, 'Please use `keras.layers.RNN(cell)`, which is equivalent to this API')
@tf_export(v1=['nn.dynamic_rnn'])
@dispatch.add_dispatch_support
def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.

  Performs fully dynamic unrolling of `inputs`.

  Example:

  ```python
  # create a BasicRNNCell
  rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(hidden_size)

  # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]

  # defining initial state
  initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

  # 'state' is a tensor of shape [batch_size, cell_state_size]
  outputs, state = tf.compat.v1.nn.dynamic_rnn(rnn_cell, input_data,
                                     initial_state=initial_state,
                                     dtype=tf.float32)
  ```

  ```python
  # create 2 LSTMCells
  rnn_layers = [tf.compat.v1.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

  # create a RNN cell composed sequentially of a number of RNNCells
  multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_layers)

  # 'outputs' is a tensor of shape [batch_size, max_time, 256]
  # 'state' is a N-tuple where N is the number of LSTMCells containing a
  # tf.nn.rnn_cell.LSTMStateTuple for each cell
  outputs, state = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell,
                                     inputs=data,
                                     dtype=tf.float32)
  ```


  Args:
    cell: An instance of RNNCell.
    inputs: The RNN inputs.
      If `time_major == False` (default), this must be a `Tensor` of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such elements.
      If `time_major == True`, this must be a `Tensor` of shape: `[max_time,
        batch_size, ...]`, or a nested tuple of such elements. This may also be
        a (possibly nested) tuple of Tensors satisfying this property.  The
        first two dimensions must match across all the inputs, but otherwise the
        ranks and other shape components may differ. In this case, input to
        `cell` at each time-step will replicate the structure of these tuples,
        except for the time dimension (from which the time is taken). The input
        to `cell` at each time step will be a `Tensor` or (possibly nested)
        tuple of Tensors each with dimensions `[batch_size, ...]`.
    sequence_length: (optional) An int32/int64 vector sized `[batch_size]`. Used
      to copy-through state and zero-out outputs when past a batch element's
      sequence length.  This parameter enables users to extract the last valid
      state and properly padded outputs, so it is provided for correctness.
    initial_state: (optional) An initial state for the RNN. If `cell.state_size`
      is an integer, this must be a `Tensor` of appropriate type and shape
      `[batch_size, cell.state_size]`. If `cell.state_size` is a tuple, this
      should be a tuple of tensors having shapes `[batch_size, s] for s in
      cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency and
      can be run in parallel, will be.  This parameter trades off time for
      space.  Values >> 1 use more memory but take less time, while smaller
      values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs which
      would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors. If true,
      these `Tensors` must be shaped `[max_time, batch_size, depth]`. If false,
      these `Tensors` must be shaped `[batch_size, max_time, depth]`. Using
      `time_major = True` is a bit more efficient because it avoids transposes
      at the beginning and end of the RNN calculation.  However, most TensorFlow
      data is batch-major, so by default this function accepts input and emits
      output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to "rnn".

  Returns:
    A pair (outputs, state) where:

    outputs: The RNN output `Tensor`.

      If time_major == False (default), this will be a `Tensor` shaped:
        `[batch_size, max_time, cell.output_size]`.

      If time_major == True, this will be a `Tensor` shaped:
        `[max_time, batch_size, cell.output_size]`.

      Note, if `cell.output_size` is a (possibly nested) tuple of integers
      or `TensorShape` objects, then `outputs` will be a tuple having the
      same structure as `cell.output_size`, containing Tensors having shapes
      corresponding to the shape data in `cell.output_size`.

    state: The final state.  If `cell.state_size` is an int, this
      will be shaped `[batch_size, cell.state_size]`.  If it is a
      `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
      If it is a (possibly nested) tuple of ints or `TensorShape`, this will
      be a tuple having the corresponding shapes. If cells are `LSTMCells`
      `state` will be a tuple containing a `LSTMStateTuple` for each cell.

  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.

  @compatibility(TF2)
  `tf.compat.v1.nn.dynamic_rnn` is not compatible with eager execution and
  `tf.function`. Please use `tf.keras.layers.RNN` instead for TF2 migration.
  Take LSTM as an example, you can instantiate a `tf.keras.layers.RNN` layer
  with `tf.keras.layers.LSTMCell`, or directly via `tf.keras.layers.LSTM`. Once
  the keras layer is created, you can get the output and states by calling
  the layer with input and states. Please refer to [this
  guide](https://www.tensorflow.org/guide/keras/rnn) for more details about
  Keras RNN. You can also find more details about the difference and comparison
  between Keras RNN and TF compat v1 rnn in [this
  document](https://github.com/tensorflow/community/blob/master/rfcs/20180920-unify-rnn-interface.md)

  #### Structural Mapping to Native TF2

  Before:

  ```python
  # create 2 LSTMCells
  rnn_layers = [tf.compat.v1.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

  # create a RNN cell composed sequentially of a number of RNNCells
  multi_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_layers)

  # 'outputs' is a tensor of shape [batch_size, max_time, 256]
  # 'state' is a N-tuple where N is the number of LSTMCells containing a
  # tf.nn.rnn_cell.LSTMStateTuple for each cell
  outputs, state = tf.compat.v1.nn.dynamic_rnn(cell=multi_rnn_cell,
                                               inputs=data,
                                               dtype=tf.float32)
  ```

  After:

  ```python
  # RNN layer can take a list of cells, which will then stack them together.
  # By default, keras RNN will only return the last timestep output and will not
  # return states. If you need whole time sequence output as well as the states,
  # you can set `return_sequences` and `return_state` to True.
  rnn_layer = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(128),
                                   tf.keras.layers.LSTMCell(256)],
                                  return_sequences=True,
                                  return_state=True)
  outputs, output_states = rnn_layer(inputs, states)
  ```

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name    | Note                             |
  | :-------------------- | :-------------- | :------------------------------- |
  | `cell`                | `cell`          | In the RNN layer constructor     |
  | `inputs`              | `inputs`        | In the RNN layer `__call__`      |
  | `sequence_length`     | Not used        | Adding masking layer before RNN  :
  :                       :                 : to achieve the same result.      :
  | `initial_state`       | `initial_state` | In the RNN layer `__call__`      |
  | `dtype`               | `dtype`         | In the RNN layer constructor     |
  | `parallel_iterations` | Not supported   |                                  |
  | `swap_memory`         | Not supported   |                                  |
  | `time_major`          | `time_major`    | In the RNN layer constructor     |
  | `scope`               | Not supported   |                                  |
  @end_compatibility
  """
    rnn_cell_impl.assert_like_rnncell('cell', cell)
    with vs.variable_scope(scope or 'rnn') as varscope:
        if _should_cache():
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)
        flat_input = nest.flatten(inputs)
        if not time_major:
            flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
            flat_input = tuple((_transpose_batch_time(input_) for input_ in flat_input))
        parallel_iterations = parallel_iterations or 32
        if sequence_length is not None:
            sequence_length = math_ops.cast(sequence_length, dtypes.int32)
            if sequence_length.get_shape().rank not in (None, 1):
                raise ValueError(f'Argument sequence_length must be a vector of length batch_size. Received sequence_length={sequence_length} of shape: {sequence_length.get_shape()}')
            sequence_length = array_ops.identity(sequence_length, name='sequence_length')
        batch_size = _best_effort_input_batch_size(flat_input)
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError('If no initial_state is provided, argument `dtype` must be specified')
            if getattr(cell, 'get_initial_state', None) is not None:
                state = cell.get_initial_state(inputs=None, batch_size=batch_size, dtype=dtype)
            else:
                state = cell.zero_state(batch_size, dtype)

        def _assert_has_shape(x, shape):
            x_shape = array_ops.shape(x)
            packed_shape = array_ops_stack.stack(shape)
            return control_flow_assert.Assert(math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)), ['Expected shape for Tensor %s is ' % x.name, packed_shape, ' but saw shape: ', x_shape])
        if not context.executing_eagerly() and sequence_length is not None:
            with ops.control_dependencies([_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = array_ops.identity(sequence_length, name='CheckSeqLen')
        inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)
        outputs, final_state = _dynamic_rnn_loop(cell, inputs, state, parallel_iterations=parallel_iterations, swap_memory=swap_memory, sequence_length=sequence_length, dtype=dtype)
        if not time_major:
            outputs = nest.map_structure(_transpose_batch_time, outputs)
        return (outputs, final_state)