from typing import Any, List, Optional, Tuple, Union
import numpy as np
import gym
from gym.vector.utils.spaces import batch_space
def reset_async(self, **kwargs):
    return self.env.reset_async(**kwargs)