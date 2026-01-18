from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils import EzPickle
import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
def reset_model(self):
    qpos = self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
    qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
    self.set_state(qpos, qvel)
    obs = self._get_obs()
    return obs