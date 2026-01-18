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
def symlog(x: 'tf.Tensor') -> 'tf.Tensor':
    """The symlog function as described in [1]:

    [1] Mastering Diverse Domains through World Models - 2023
    D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
    https://arxiv.org/pdf/2301.04104v1.pdf
    """
    return tf.math.sign(x) * tf.math.log(tf.math.abs(x) + 1)