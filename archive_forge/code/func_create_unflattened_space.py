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
def create_unflattened_space(self):
    return Dict({k: v.unflattened if hasattr(v, 'unflattened') else v for k, v in self.spaces.items() if not v.is_flattenable()})