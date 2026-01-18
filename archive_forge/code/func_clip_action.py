import gymnasium as gym
from gymnasium.spaces import Tuple, Dict
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
import tree  # pip install dm_tree
from typing import Any, List, Optional, Union
@DeveloperAPI
def clip_action(action, action_space):
    """Clips all components in `action` according to the given Space.

    Only applies to Box components within the action space.

    Args:
        action: The action to be clipped. This could be any complex
            action, e.g. a dict or tuple.
        action_space: The action space struct,
            e.g. `{"a": Distrete(2)}` for a space: Dict({"a": Discrete(2)}).

    Returns:
        Any: The input action, but clipped by value according to the space's
            bounds.
    """

    def map_(a, s):
        if isinstance(s, gym.spaces.Box):
            a = np.clip(a, s.low, s.high)
        return a
    return tree.map_structure(map_, action, action_space)