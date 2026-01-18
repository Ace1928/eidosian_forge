from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Union
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.images import rgb2gray, resize
def next_episode_results(self):
    for i in range(self._num_returned, len(self._episode_rewards)):
        yield (self._episode_rewards[i], self._episode_lengths[i])
    self._num_returned = len(self._episode_rewards)