import gymnasium as gym
from ray.rllib.utils.annotations import PublicAPI
def assertSpace(self, space):
    err = 'Values of the dict should be instances of gym.Space'
    assert issubclass(type(space), gym.spaces.Space), err