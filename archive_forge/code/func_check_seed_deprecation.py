import inspect
from copy import deepcopy
import numpy as np
import gym
from gym import logger, spaces
from gym.utils.passive_env_checker import (
def check_seed_deprecation(env: gym.Env):
    """Makes sure support for deprecated function `seed` is dropped.

    Args:
        env: The environment to check
    Raises:
        UserWarning
    """
    seed_fn = getattr(env, 'seed', None)
    if callable(seed_fn):
        logger.warn('Official support for the `seed` function is dropped. Standard practice is to reset gym environments using `env.reset(seed=<desired seed>)`')