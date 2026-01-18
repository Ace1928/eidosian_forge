import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
def softplus_std_fn(std_for_zero_param: float, min_std: float):
    std_shift = tfp.math.softplus_inverse(std_for_zero_param - min_std)
    fn = lambda scale: tf.nn.softplus(scale + std_shift) + min_std
    assert abs(fn(0) - std_for_zero_param) < 0.001
    return fn