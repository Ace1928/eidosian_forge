import collections
import pickle
import threading
import time
import timeit
from absl import flags
from absl import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.distribute import values as values_lib  
from tensorflow.python.framework import composite_tensor  
from tensorflow.python.framework import tensor_conversion_registry  
def batch_apply(fn, inputs):
    """Folds time into the batch dimension, runs fn() and unfolds the result.

  Args:
    fn: Function that takes as input the n tensors of the tf.nest structure,
      with shape [time*batch, <remaining shape>], and returns a tf.nest
      structure of batched tensors.
    inputs: tf.nest structure of n [time, batch, <remaining shape>] tensors.

  Returns:
    tf.nest structure of [time, batch, <fn output shape>]. Structure is
    determined by the output of fn.
  """
    time_to_batch_fn = lambda t: tf.reshape(t, [-1] + t.shape[2:].as_list())
    batched = tf.nest.map_structure(time_to_batch_fn, inputs)
    output = fn(*batched)
    prefix = [int(tf.nest.flatten(inputs)[0].shape[0]), -1]
    batch_to_time_fn = lambda t: tf.reshape(t, prefix + t.shape[1:].as_list())
    return tf.nest.map_structure(batch_to_time_fn, output)