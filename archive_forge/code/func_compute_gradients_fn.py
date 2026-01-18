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
def compute_gradients_fn(policy: Policy, optimizer: LocalOptimizer, loss: TensorType) -> ModelGradients:
    grads_and_vars = sac_compute_and_clip_gradients(policy, optimizer, loss)
    if policy.config['lagrangian']:
        if policy.config['framework'] == 'tf2':
            tape = optimizer.tape
            log_alpha_prime = [policy.model.log_alpha_prime]
            alpha_prime_grads_and_vars = list(zip(tape.gradient(policy.alpha_prime_loss, log_alpha_prime), log_alpha_prime))
        else:
            alpha_prime_grads_and_vars = policy._alpha_prime_optimizer.compute_gradients(policy.alpha_prime_loss, var_list=[policy.model.log_alpha_prime])
        if policy.config['grad_clip']:
            clip_func = partial(tf.clip_by_norm, clip_norm=policy.config['grad_clip'])
        else:
            clip_func = tf.identity
        policy._alpha_prime_grads_and_vars = [(clip_func(g), v) for g, v in alpha_prime_grads_and_vars if g is not None]
        grads_and_vars += policy._alpha_prime_grads_and_vars
    return grads_and_vars