import abc
from typing import Callable
import dataclasses
import gym
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler
def check_box_space(space):
    assert len(space.shape) == 1, space.shape
    if any((l != -1 for l in space.low)):
        raise ValueError(f'Learner only supports actions bounded to [-1,1]: {space.low}')
    if any((h != 1 for h in space.high)):
        raise ValueError(f'Learner only supports actions bounded to [-1,1]: {space.high}')