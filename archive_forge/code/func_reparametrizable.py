import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
@property
def reparametrizable(self):
    return self._create_dist(tf.zeros((self._param_size,))).reparameterization_type == tfd.FULLY_REPARAMETERIZED