import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.utils.annotations import PublicAPI
@property
def observation_space(self):
    spec = self._env.observation_spec()
    return _convert_spec_to_space(spec)