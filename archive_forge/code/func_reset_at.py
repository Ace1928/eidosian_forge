import gymnasium as gym
import numpy as np
from typing import Optional
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.annotations import override
@override(VectorEnv)
def reset_at(self, index, *, seed=None, options=None):
    self.ts = 0
    return self.env.reset(seed=seed, options=options)