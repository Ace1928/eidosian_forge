from os import path
from typing import Optional, Union
import numpy as np
import gym
from gym import error, logger, spaces
from gym.spaces import Space
def get_body_com(self, body_name):
    return self.data.body(body_name).xpos