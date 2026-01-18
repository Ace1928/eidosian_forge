import logging
import os
import warnings
from typing import Dict, List, Optional, TYPE_CHECKING, Union
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete
from packaging import version
import ray
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.utils.annotations import Deprecated, PublicAPI, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import SMALL_NUMBER
from ray.rllib.utils.typing import (
@PublicAPI
def copy_torch_tensors(x: TensorStructType, device: Optional[str]=None):
    """Creates a copy of `x` and makes deep copies torch.Tensors in x.

    Also moves the copied tensors to the specified device (if not None).

    Note if an object in x is not a torch.Tensor, it will be shallow-copied.

    Args:
        x : Any (possibly nested) struct possibly containing torch.Tensors.
        device : The device to move the tensors to.

    Returns:
        Any: A new struct with the same structure as `x`, but with all
            torch.Tensors deep-copied and moved to the specified device.

    """

    def mapping(item):
        if isinstance(item, torch.Tensor):
            return torch.clone(item.detach()) if device is None else item.detach().to(device)
        else:
            return item
    return tree.map_structure(mapping, x)