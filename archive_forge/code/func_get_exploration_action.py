import functools
import gymnasium as gym
import numpy as np
from typing import Optional, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import (
from ray.rllib.utils.tf_utils import zero_logps_from_actions
@override(Exploration)
def get_exploration_action(self, *, action_distribution: ActionDistribution, timestep: Optional[Union[int, TensorType]]=None, explore: bool=True):
    if self.framework == 'torch':
        return self._get_torch_exploration_action(action_distribution, timestep, explore)
    else:
        return self._get_tf_exploration_action_op(action_distribution, timestep, explore)