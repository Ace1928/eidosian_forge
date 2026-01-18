from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium.utils import EzPickle
import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
def set_task(self, task):
    """
        Args:
            task: Task of the meta-learning environment (here: mass of
                the pendulum).
        """
    self.m = task