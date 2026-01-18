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
def split_structure(structure, prefix_length, axis=0):
    """Splits in two a tf.nest structure of tensors along the first axis."""
    flattened = tf.nest.flatten(structure)
    split = [tf.split(x, [prefix_length, tf.shape(x)[axis] - prefix_length], axis=axis) for x in flattened]
    flattened_prefix = [pair[0] for pair in split]
    flattened_suffix = [pair[1] for pair in split]
    return (tf.nest.pack_sequence_as(structure, flattened_prefix), tf.nest.pack_sequence_as(structure, flattened_suffix))