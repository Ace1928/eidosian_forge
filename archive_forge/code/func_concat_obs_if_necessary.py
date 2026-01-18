import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Optional
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType
def concat_obs_if_necessary(obs: TensorStructType):
    """Concat model outs if they come as original tuple observations."""
    if isinstance(obs, (list, tuple)):
        obs = torch.cat(obs, dim=-1)
    elif isinstance(obs, dict):
        obs = torch.cat([torch.unsqueeze(val, 1) if len(val.shape) == 1 else val for val in tree.flatten(obs.values())], dim=-1)
    return obs