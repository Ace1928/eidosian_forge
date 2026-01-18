import logging
from typing import Any, Callable, List, Optional, Type, TYPE_CHECKING, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete
from ray.rllib.utils.annotations import PublicAPI, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import (
@PublicAPI
def reduce_mean_ignore_inf(x: TensorType, axis: Optional[int]=None) -> TensorType:
    """Same as tf.reduce_mean() but ignores -inf values.

    Args:
        x: The input tensor to reduce mean over.
        axis: The axis over which to reduce. None for all axes.

    Returns:
        The mean reduced inputs, ignoring inf values.
    """
    mask = tf.not_equal(x, tf.float32.min)
    x_zeroed = tf.where(mask, x, tf.zeros_like(x))
    return tf.math.reduce_sum(x_zeroed, axis) / tf.math.reduce_sum(tf.cast(mask, tf.float32), axis)