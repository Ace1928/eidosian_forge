from functools import partial
import numpy as np
import gymnasium as gym
import logging
import tree
from typing import Dict, List, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.sac.sac_tf_policy import (
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.tf_mixins import TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import get_variable, try_import_tf, try_import_tfp
from ray.rllib.utils.typing import (
def cql_stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    sac_dict = stats(policy, train_batch)
    sac_dict['cql_loss'] = tf.reduce_mean(tf.stack(policy.cql_loss))
    if policy.config['lagrangian']:
        sac_dict['log_alpha_prime_value'] = policy.log_alpha_prime_value
        sac_dict['alpha_prime_value'] = policy.alpha_prime_value
        sac_dict['alpha_prime_loss'] = policy.alpha_prime_loss
    return sac_dict