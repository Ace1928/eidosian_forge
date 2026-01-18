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
def set_torch_seed(seed: Optional[int]=None) -> None:
    """Sets the torch random seed to the given value.

    Args:
        seed: The seed to use or None for no seeding.
    """
    if seed is not None and torch:
        torch.manual_seed(seed)
        cuda_version = torch.version.cuda
        if cuda_version is not None and float(torch.version.cuda) >= 10.2:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = '4096:8'
        else:
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True