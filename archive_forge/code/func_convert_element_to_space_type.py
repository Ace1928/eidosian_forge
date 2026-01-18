import gymnasium as gym
from gymnasium.spaces import Tuple, Dict
import numpy as np
from ray.rllib.utils.annotations import DeveloperAPI
import tree  # pip install dm_tree
from typing import Any, List, Optional, Union
@DeveloperAPI
def convert_element_to_space_type(element: Any, sampled_element: Any) -> Any:
    """Convert all the components of the element to match the space dtypes.

    Args:
        element: The element to be converted.
        sampled_element: An element sampled from a space to be matched
            to.

    Returns:
        The input element, but with all its components converted to match
        the space dtypes.
    """

    def map_(elem, s):
        if isinstance(s, np.ndarray):
            if not isinstance(elem, np.ndarray):
                assert isinstance(elem, (float, int)), f'ERROR: `elem` ({elem}) must be np.array, float or int!'
                if s.shape == ():
                    elem = np.array(elem, dtype=s.dtype)
                else:
                    raise ValueError('Element should be of type np.ndarray but is instead of                             type {}'.format(type(elem)))
            elif s.dtype != elem.dtype:
                elem = elem.astype(s.dtype)
        elif isinstance(s, int) or isinstance(s, np.int_):
            if isinstance(elem, float) and elem.is_integer():
                elem = int(elem)
            if isinstance(elem, np.float_):
                elem = np.int64(elem)
        return elem
    return tree.map_structure(map_, element, sampled_element, check_types=False)