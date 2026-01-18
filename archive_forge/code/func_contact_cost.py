import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
@property
def contact_cost(self):
    contact_cost = self._contact_cost_weight * np.sum(np.square(self.contact_forces))
    return contact_cost