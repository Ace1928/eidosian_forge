import gymnasium as gym
from gymnasium.spaces import Tuple, Dict
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
import tree  # pip install dm_tree
from typing import Any, List, Optional, Union
@DeveloperAPI
def get_original_space(space: gym.Space) -> gym.Space:
    """Returns the original space of a space, if any.

    This function recursively traverses the given space and returns the original space
    at the very end of the chain.

    Args:
        space: The space to get the original space for.

    Returns:
        The original space or the given space itself if no original space is found.
    """
    if hasattr(space, 'original_space'):
        return get_original_space(space.original_space)
    else:
        return space