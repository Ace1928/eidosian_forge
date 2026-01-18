import gymnasium as gym
import numpy as np
from typing import Optional, List, Mapping, Iterable, Dict
import tree
import abc
from ray.rllib.models.distributions import Distribution
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, Union, Tuple
@staticmethod
@override(Distribution)
def required_input_dim(space: gym.Space, input_lens: List[int], **kwargs) -> int:
    return sum(input_lens)