import inspect
from functools import partial
from typing import Callable
import numpy as np
from gym import Space, error, logger, spaces
def check_obs(obs, observation_space: spaces.Space, method_name: str):
    """Check that the observation returned by the environment correspond to the declared one.

    Args:
        obs: The observation to check
        observation_space: The observation space of the observation
        method_name: The method name that generated the observation
    """
    pre = f'The obs returned by the `{method_name}()` method'
    if isinstance(observation_space, spaces.Discrete):
        if not isinstance(obs, (np.int64, int)):
            logger.warn(f'{pre} should be an int or np.int64, actual type: {type(obs)}')
    elif isinstance(observation_space, spaces.Box):
        if observation_space.shape != ():
            if not isinstance(obs, np.ndarray):
                logger.warn(f'{pre} was expecting a numpy array, actual type: {type(obs)}')
            elif obs.dtype != observation_space.dtype:
                logger.warn(f'{pre} was expecting numpy array dtype to be {observation_space.dtype}, actual type: {obs.dtype}')
    elif isinstance(observation_space, (spaces.MultiBinary, spaces.MultiDiscrete)):
        if not isinstance(obs, np.ndarray):
            logger.warn(f'{pre} was expecting a numpy array, actual type: {type(obs)}')
    elif isinstance(observation_space, spaces.Tuple):
        if not isinstance(obs, tuple):
            logger.warn(f'{pre} was expecting a tuple, actual type: {type(obs)}')
        assert len(obs) == len(observation_space.spaces), f'{pre} length is not same as the observation space length, obs length: {len(obs)}, space length: {len(observation_space.spaces)}'
        for sub_obs, sub_space in zip(obs, observation_space.spaces):
            check_obs(sub_obs, sub_space, method_name)
    elif isinstance(observation_space, spaces.Dict):
        assert isinstance(obs, dict), f'{pre} must be a dict, actual type: {type(obs)}'
        assert obs.keys() == observation_space.spaces.keys(), f'{pre} observation keys is not same as the observation space keys, obs keys: {list(obs.keys())}, space keys: {list(observation_space.spaces.keys())}'
        for space_key in observation_space.spaces.keys():
            check_obs(obs[space_key], observation_space[space_key], method_name)
    try:
        if obs not in observation_space:
            logger.warn(f'{pre} is not within the observation space.')
    except Exception as e:
        logger.warn(f'{pre} is not within the observation space with exception: {e}')