import inspect
from copy import deepcopy
import numpy as np
import gym
from gym import logger, spaces
from gym.utils.passive_env_checker import (
def check_space_limit(space, space_type: str):
    """Check the space limit for only the Box space as a test that only runs as part of `check_env`."""
    if isinstance(space, spaces.Box):
        if np.any(np.equal(space.low, -np.inf)):
            logger.warn(f'A Box {space_type} space minimum value is -infinity. This is probably too low.')
        if np.any(np.equal(space.high, np.inf)):
            logger.warn(f'A Box {space_type} space maximum value is -infinity. This is probably too high.')
        if space_type == 'action':
            if len(space.shape) == 1:
                if np.any(np.logical_and(space.low != np.zeros_like(space.low), np.abs(space.low) != np.abs(space.high))) or np.any(space.low < -1) or np.any(space.high > 1):
                    logger.warn('For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.')
    elif isinstance(space, spaces.Tuple):
        for subspace in space.spaces:
            check_space_limit(subspace, space_type)
    elif isinstance(space, spaces.Dict):
        for subspace in space.values():
            check_space_limit(subspace, space_type)