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
def tensor_spec_from_gym_space(space, name):
    """Get a TensorSpec from a gym spec."""
    if space.shape is not None:
        return tf.TensorSpec(space.shape, space.dtype, name)
    if not isinstance(space, gym.spaces.Tuple):
        raise ValueError("Space '{}' is not a tuple: unknown shape.".format(space))
    num_elements = 0
    for s in space:
        if len(s.shape) != 1:
            raise ValueError('Only 1 dimension subspaces are handled for tuple spaces: {}'.format(space))
        num_elements += s.shape[0]
    return tf.TensorSpec((num_elements,), tf.float32, name)