import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
@tf.custom_gradient
def safe_exp(x):
    e = tf.exp(tf.clip_by_value(x, -15, 15))

    def grad(dy):
        return dy * e
    return (e, grad)