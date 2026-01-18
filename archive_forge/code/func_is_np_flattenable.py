from typing import Dict, List, Optional, Sequence, SupportsFloat, Tuple, Type, Union
import numpy as np
import gym.error
from gym import logger
from gym.spaces.space import Space
@property
def is_np_flattenable(self):
    """Checks whether this space can be flattened to a :class:`spaces.Box`."""
    return True