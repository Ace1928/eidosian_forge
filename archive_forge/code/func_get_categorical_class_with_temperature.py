import functools
import gymnasium as gym
from math import log
import numpy as np
import tree  # pip install dm_tree
from typing import Optional
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils import MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT, SMALL_NUMBER
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
@DeveloperAPI
def get_categorical_class_with_temperature(t: float):
    """Categorical distribution class that has customized default temperature."""

    class CategoricalWithTemperature(Categorical):

        def __init__(self, inputs, model=None, temperature=t):
            super().__init__(inputs, model, temperature)
    return CategoricalWithTemperature