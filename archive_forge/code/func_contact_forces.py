import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
@property
def contact_forces(self):
    raw_contact_forces = self.data.cfrc_ext
    min_value, max_value = self._contact_force_range
    contact_forces = np.clip(raw_contact_forces, min_value, max_value)
    return contact_forces