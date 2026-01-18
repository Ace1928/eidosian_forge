from collections import OrderedDict
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import tree  # pip install dm_tree
from types import MappingProxyType
from typing import List, Optional
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import SpaceStruct, TensorType, TensorStructType, Union
@PublicAPI
def one_hot_multidiscrete(x, depths=List[int]):
    if torch and isinstance(x, torch.Tensor):
        x = x.numpy()
    shape = x.shape
    return np.concatenate([one_hot(x[i] if len(shape) == 1 else x[:, i], depth=n).astype(np.float32) for i, n in enumerate(depths)], axis=-1)