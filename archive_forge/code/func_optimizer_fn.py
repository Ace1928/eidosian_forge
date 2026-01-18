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
def optimizer_fn(policy: Policy, config: AlgorithmConfigDict) -> Tuple[LocalOptimizer]:
    """Creates all necessary optimizers for SAC learning.

    The 3 or 4 (twin_q=True) optimizers returned here correspond to the
    number of loss terms returned by the loss function.

    Args:
        policy: The policy object to be trained.
        config: The Algorithm's config dict.

    Returns:
        Tuple[LocalOptimizer]: The local optimizers to use for policy training.
    """
    policy.actor_optim = torch.optim.Adam(params=policy.model.policy_variables(), lr=config['optimization']['actor_learning_rate'], eps=1e-07)
    critic_split = len(policy.model.q_variables())
    if config['twin_q']:
        critic_split //= 2
    policy.critic_optims = [torch.optim.Adam(params=policy.model.q_variables()[:critic_split], lr=config['optimization']['critic_learning_rate'], eps=1e-07)]
    if config['twin_q']:
        policy.critic_optims.append(torch.optim.Adam(params=policy.model.q_variables()[critic_split:], lr=config['optimization']['critic_learning_rate'], eps=1e-07))
    policy.alpha_optim = torch.optim.Adam(params=[policy.model.log_alpha], lr=config['optimization']['entropy_learning_rate'], eps=1e-07)
    return tuple([policy.actor_optim] + policy.critic_optims + [policy.alpha_optim])