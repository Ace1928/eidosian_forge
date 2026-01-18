from typing import Any, List, Optional, Tuple, Union
import numpy as np
import gym
from gym.vector.utils.spaces import batch_space
def call_async(self, name, *args, **kwargs):
    """Calls a method name for each parallel environment asynchronously."""