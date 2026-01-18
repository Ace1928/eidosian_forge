import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import logging
import tree  # pip install dm_tree
from typing import Dict, List, Optional, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.sac.sac_tf_policy import (
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.policy.torch_mixins import TargetNetworkMixin
from ray.rllib.utils.torch_utils import (
from ray.rllib.utils.typing import (
def setup_late_mixins(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> None:
    """Call mixin classes' constructors after Policy initialization.

    - Moves the target model(s) to the GPU, if necessary.
    - Adds the `compute_td_error` method to the given policy.
    Calling `compute_td_error` with batch data will re-calculate the loss
    on that batch AND return the per-batch-item TD-error for prioritized
    replay buffer record weight updating (in case a prioritized replay buffer
    is used).
    - Also adds the `update_target` method to the given policy.
    Calling `update_target` updates all target Q-networks' weights from their
    respective "main" Q-metworks, based on tau (smooth, partial updating).

    Args:
        policy: The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config: The Policy's config.
    """
    ComputeTDErrorMixin.__init__(policy)
    TargetNetworkMixin.__init__(policy)