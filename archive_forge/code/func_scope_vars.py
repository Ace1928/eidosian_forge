import logging
from typing import Any, Callable, List, Optional, Type, TYPE_CHECKING, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete
from ray.rllib.utils.annotations import PublicAPI, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import (
@PublicAPI
def scope_vars(scope: Union[str, 'tf1.VariableScope'], trainable_only: bool=False) -> List['tf.Variable']:
    """Get variables inside a given scope.

    Args:
        scope: Scope in which the variables reside.
        trainable_only: Whether or not to return only the variables that were
            marked as trainable.

    Returns:
        The list of variables in the given `scope`.
    """
    return tf1.get_collection(tf1.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf1.GraphKeys.VARIABLES, scope=scope if isinstance(scope, str) else scope.name)