from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union
import numpy as np
import gym.error
from gym import logger
from gym.spaces.space import Space
def is_float_integer(var) -> bool:
    """Checks if a variable is an integer or float."""
    return np.issubdtype(type(var), np.integer) or np.issubdtype(type(var), np.floating)