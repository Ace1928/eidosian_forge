from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Union
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.images import rgb2gray, resize
@PublicAPI
def is_atari(env: Union[gym.Env, str]) -> bool:
    """Returns, whether a given env object or env descriptor (str) is an Atari env.

    Args:
        env: The gym.Env object or a string descriptor of the env (e.g. "ALE/Pong-v5").

    Returns:
        Whether `env` is an Atari environment.
    """
    if not isinstance(env, str):
        if hasattr(env.observation_space, 'shape') and env.observation_space.shape is not None and (len(env.observation_space.shape) <= 2):
            return False
        return 'AtariEnv<ALE' in str(env)
    else:
        return env.startswith('ALE/')