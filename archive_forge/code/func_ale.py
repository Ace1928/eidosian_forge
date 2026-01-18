import numpy as np
import gym
from gym.spaces import Box
@property
def ale(self):
    """Make ale as a class property to avoid serialization error."""
    return self.env.unwrapped.ale