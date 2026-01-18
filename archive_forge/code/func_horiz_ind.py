import math
from typing import Optional, Union
import numpy as np
import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.error import DependencyNotInstalled, InvalidAction
from gym.utils import EzPickle
def horiz_ind(place, val):
    return [((place + 0) * s, H - 4 * h), ((place + val) * s, H - 4 * h), ((place + val) * s, H - 2 * h), ((place + 0) * s, H - 2 * h)]