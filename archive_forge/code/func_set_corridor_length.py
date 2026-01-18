import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
def set_corridor_length(self, length):
    self.end_pos = length
    print('Updated corridor length to {}'.format(length))