import uuid
import tensorflow.compat.v2 as tf
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine import base_layer
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.rnn import gru_lstm_utils
from keras.src.layers.rnn import rnn_utils
from keras.src.layers.rnn.base_rnn import RNN
from keras.src.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def standard_gru(inputs, init_h, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, sequence_lengths, zero_output_for_mask, return_sequences):
    """GRU with standard kernel implementation.

    This implementation can be run on all types of hardware.

    This implementation lifts out all the layer weights and make them function
    parameters. It has same number of tensor input params as the cuDNN
    counterpart. The RNN step logic has been simplified, eg dropout and mask is
    removed since cuDNN implementation does not support that.

    Args:
      inputs: Input tensor of GRU layer.
      init_h: Initial state tensor for the cell output.
      kernel: Weights for cell kernel.
      recurrent_kernel: Weights for cell recurrent kernel.
      bias: Weights for cell kernel bias and recurrent bias. The bias contains
        the combined input_bias and recurrent_bias.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked. An individual `True` entry indicates
        that the corresponding timestep should be utilized, while a `False`
        entry indicates that the corresponding timestep should be ignored.
      time_major: Boolean, whether the inputs are in the format of
        [time, batch, feature] or [batch, time, feature].
      go_backwards: Boolean (default False). If True, process the input sequence
        backwards and return the reversed sequence.
      sequence_lengths: The lengths of all sequences coming from a variable
        length input, such as ragged tensors. If the input has a fixed timestep
        size, this should be None.
      zero_output_for_mask: Boolean, whether to output zero for masked timestep.
      return_sequences: Boolean. If True, return the recurrent outputs for all
        timesteps in the sequence. If False, only return the output for the
        last timestep (which consumes less memory).

    Returns:
      last_output: output tensor for the last timestep, which has shape
        [batch, units].
      outputs:
        - If `return_sequences=True`: output tensor for all timesteps,
          which has shape [batch, time, units].
        - Else, a tensor equal to `last_output` with shape [batch, 1, units]
      state_0: the cell output, which has same shape as init_h.
      runtime: constant string tensor which indicate real runtime hardware. This
        value is for testing purpose and should be used by user.
    """
    input_shape = backend.int_shape(inputs)
    timesteps = input_shape[0] if time_major else input_shape[1]
    input_bias, recurrent_bias = tf.unstack(bias)

    def step(cell_inputs, cell_states):
        """Step function that will be used by Keras RNN backend."""
        h_tm1 = cell_states[0]
        matrix_x = backend.dot(cell_inputs, kernel)
        matrix_x = backend.bias_add(matrix_x, input_bias)
        x_z, x_r, x_h = tf.split(matrix_x, 3, axis=1)
        matrix_inner = backend.dot(h_tm1, recurrent_kernel)
        matrix_inner = backend.bias_add(matrix_inner, recurrent_bias)
        recurrent_z, recurrent_r, recurrent_h = tf.split(matrix_inner, 3, axis=1)
        z = tf.sigmoid(x_z + recurrent_z)
        r = tf.sigmoid(x_r + recurrent_r)
        hh = tf.tanh(x_h + r * recurrent_h)
        h = z * h_tm1 + (1 - z) * hh
        return (h, [h])
    last_output, outputs, new_states = backend.rnn(step, inputs, [init_h], constants=None, unroll=False, time_major=time_major, mask=mask, go_backwards=go_backwards, input_length=sequence_lengths if sequence_lengths is not None else timesteps, zero_output_for_mask=zero_output_for_mask, return_all_outputs=return_sequences)
    return (last_output, outputs, new_states[0], gru_lstm_utils.runtime(gru_lstm_utils.RUNTIME_CPU))