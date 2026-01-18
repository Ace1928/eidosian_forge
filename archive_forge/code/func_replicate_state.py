from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def replicate_state(start_state, batch_size):
    """Create batch versions of state.

  Takes a list of Tensors, adds a batch dimension, and replicates
  batch_size times across that batch dimension. Used to replicate the
  non-batch state returned by get_start_state in define_loss.

  Args:
    start_state: Model-defined state to replicate.
    batch_size: Batch dimension for data.

  Returns:
    Replicated versions of the state.
  """
    flattened_state = tf.nest.flatten(start_state)
    replicated_state = [tf.tile(tf.compat.v1.expand_dims(state_nonbatch, 0), tf.concat([[batch_size], tf.ones([tf.rank(state_nonbatch)], dtype=tf.dtypes.int32)], 0)) for state_nonbatch in flattened_state]
    return tf.nest.pack_sequence_as(start_state, replicated_state)