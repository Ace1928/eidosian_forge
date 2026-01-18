from functools import partial
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple
import logging
import numpy as np
import tree  # pip install dm_tree
from typing import List, Optional, Type, Union
from ray.tune.registry import (
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor, Preprocessor
from ray.rllib.models.tf.tf_action_dist import (
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.spaces.space_utils import flatten_space
from ray.rllib.utils.typing import ModelConfigDict, TensorType
@staticmethod
@DeveloperAPI
def get_action_placeholder(action_space: gym.Space, name: str='action') -> TensorType:
    """Returns an action placeholder consistent with the action space

        Args:
            action_space: Action space of the target gym env.
            name: An optional string to name the placeholder by.
                Default: "action".

        Returns:
            action_placeholder: A placeholder for the actions
        """
    dtype, shape = ModelCatalog.get_action_shape(action_space, framework='tf')
    return tf1.placeholder(dtype, shape=shape, name=name)