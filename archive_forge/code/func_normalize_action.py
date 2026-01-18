import gymnasium as gym
from gymnasium.spaces import Tuple, Dict
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
import tree  # pip install dm_tree
from typing import Any, List, Optional, Union
@DeveloperAPI
def normalize_action(action, action_space_struct):
    """Normalizes all (Box) components in `action` to be in [-1.0, 1.0].

    Inverse of `unsquash_action()`. Useful for mapping an env's action
    (arbitrary bounded values) to a [-1.0, 1.0] interval.
    This only applies to Box components within the action space, whose
    dtype is float32 or float64.

    Args:
        action: The action to be normalized. This could be any complex
            action, e.g. a dict or tuple.
        action_space_struct: The action space struct,
            e.g. `{"a": Box()}` for a space: Dict({"a": Box()}).

    Returns:
        Any: The input action, but normalized, according to the space's
            bounds.
    """

    def map_(a, s):
        if isinstance(s, gym.spaces.Box) and (s.dtype == np.float32 or s.dtype == np.float64):
            a = (a - s.low) * 2.0 / (s.high - s.low) - 1.0
        return a
    return tree.map_structure(map_, action, action_space_struct)