import random
import string
import gym
import gym.spaces
import numpy as np
import random
from collections import OrderedDict
from typing import List
import gym
import logging
import gym.spaces
import numpy as np
import collections
import warnings
import abc
@property
def flattened(self) -> gym.spaces.Box:
    if not hasattr(self, '_flattened'):
        self._flattened = self.create_flattened_space()
    return self._flattened