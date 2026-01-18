import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()