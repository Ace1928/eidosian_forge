import time
from collections import deque
from typing import Optional
import numpy as np
import gym
def add_vector_episode_statistics(info: dict, episode_info: dict, num_envs: int, env_num: int):
    """Add episode statistics.

    Add statistics coming from the vectorized environment.

    Args:
        info (dict): info dict of the environment.
        episode_info (dict): episode statistics data.
        num_envs (int): number of environments.
        env_num (int): env number of the vectorized environments.

    Returns:
        info (dict): the input info dict with the episode statistics.
    """
    info['episode'] = info.get('episode', {})
    info['_episode'] = info.get('_episode', np.zeros(num_envs, dtype=bool))
    info['_episode'][env_num] = True
    for k in episode_info.keys():
        info_array = info['episode'].get(k, np.zeros(num_envs))
        info_array[env_num] = episode_info[k]
        info['episode'][k] = info_array
    return info