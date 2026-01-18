import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
def joint_distribution(parametric_distributions, dtype_override=tf.float32):
    """Initialize the distribution.

  Args:
    parametric_distributions: A list of ParametricDistributions.
    dtype_override: The type to output the actions in.

  Returns:
    A tuple (param size, fn(params) -> distribution)
  """
    param_sizes = [dist.param_size for dist in parametric_distributions]

    def create_dist(parameters):
        split_params = tf.split(parameters, param_sizes, axis=-1)
        dists = [dist(param) for dist, param in zip(parametric_distributions, split_params)]
        return tfd.Blockwise(dists, dtype_override=dtype_override)
    return ParametricDistribution(sum(param_sizes), create_dist)