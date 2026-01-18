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
def get_action_shape(action_space: gym.Space, framework: str='tf') -> (np.dtype, List[int]):
    """Returns action tensor dtype and shape for the action space.

        Args:
            action_space: Action space of the target gym env.
            framework: The framework identifier. One of "tf" or "torch".

        Returns:
            (dtype, shape): Dtype and shape of the actions tensor.
        """
    dl_lib = torch if framework == 'torch' else tf
    if isinstance(action_space, Discrete):
        return (action_space.dtype, (None,))
    elif isinstance(action_space, (Box, Simplex)):
        if np.issubdtype(action_space.dtype, np.floating):
            return (dl_lib.float32, (None,) + action_space.shape)
        elif np.issubdtype(action_space.dtype, np.integer):
            return (dl_lib.int32, (None,) + action_space.shape)
        else:
            raise ValueError("RLlib doesn't support non int or float box spaces")
    elif isinstance(action_space, MultiDiscrete):
        return (action_space.dtype, (None,) + action_space.shape)
    elif isinstance(action_space, (Tuple, Dict)):
        flat_action_space = flatten_space(action_space)
        size = 0
        all_discrete = True
        for i in range(len(flat_action_space)):
            if isinstance(flat_action_space[i], Discrete):
                size += 1
            else:
                all_discrete = False
                size += np.product(flat_action_space[i].shape)
        size = int(size)
        return (dl_lib.int32 if all_discrete else dl_lib.float32, (None, size))
    else:
        raise NotImplementedError('Action space {} not supported'.format(action_space))