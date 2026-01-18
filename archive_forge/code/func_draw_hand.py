import os
from typing import Optional
import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]