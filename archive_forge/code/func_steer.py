import math
import Box2D
import numpy as np
from gym.error import DependencyNotInstalled
def steer(self, s):
    """control: steer

        Args:
            s (-1..1): target position, it takes time to rotate steering wheel from side-to-side"""
    self.wheels[0].steer = s
    self.wheels[1].steer = s