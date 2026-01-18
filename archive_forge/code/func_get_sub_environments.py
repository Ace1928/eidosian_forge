import gymnasium as gym
import numpy as np
from typing import Optional
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.annotations import override
@override(VectorEnv)
def get_sub_environments(self):
    return [self.env for _ in range(self.num_envs)]