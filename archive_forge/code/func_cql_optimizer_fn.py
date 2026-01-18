import numpy as np
import gymnasium as gym
import logging
import tree
from typing import Dict, List, Tuple, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.sac.sac_tf_policy import (
from ray.rllib.algorithms.sac.sac_torch_policy import (
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_mixins import TargetNetworkMixin
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import LocalOptimizer, TensorType, AlgorithmConfigDict
from ray.rllib.utils.torch_utils import (
def cql_optimizer_fn(policy: Policy, config: AlgorithmConfigDict) -> Tuple[LocalOptimizer]:
    policy.cur_iter = 0
    opt_list = optimizer_fn(policy, config)
    if config['lagrangian']:
        log_alpha_prime = nn.Parameter(torch.zeros(1, requires_grad=True).float())
        policy.model.register_parameter('log_alpha_prime', log_alpha_prime)
        policy.alpha_prime_optim = torch.optim.Adam(params=[policy.model.log_alpha_prime], lr=config['optimization']['critic_learning_rate'], eps=1e-07)
        return tuple([policy.actor_optim] + policy.critic_optims + [policy.alpha_optim] + [policy.alpha_prime_optim])
    return opt_list