import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from functools import partial
import logging
from typing import Dict, List, Optional, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
from ray.rllib.algorithms.sac.sac_tf_model import SACTFModel
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import (
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_mixins import TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import get_variable, try_import_tf
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.tf_utils import huber_loss, make_tf_callable
from ray.rllib.utils.typing import (
def setup_mid_mixins(policy: Policy, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, config: AlgorithmConfigDict) -> None:
    """Call mixin classes' constructors before Policy's loss initialization.

    Adds the `compute_td_error` method to the given policy.
    Calling `compute_td_error` with batch data will re-calculate the loss
    on that batch AND return the per-batch-item TD-error for prioritized
    replay buffer record weight updating (in case a prioritized replay buffer
    is used).

    Args:
        policy: The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config: The Policy's config.
    """
    ComputeTDErrorMixin.__init__(policy, sac_actor_critic_loss)