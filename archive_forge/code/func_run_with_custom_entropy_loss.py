import argparse
import os
from pettingzoo.classic import rps_v2
import random
import ray
from ray import air, tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import (
from ray.rllib.env import PettingZooEnv
from ray.rllib.examples.policy.rock_paper_scissors_dummies import (
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env
def run_with_custom_entropy_loss(args, stop):
    """Example of customizing the loss function of an existing policy.

    This performs about the same as the default loss does."""
    policy_cls = {'torch': PPOTorchPolicy, 'tf': PPOTF1Policy, 'tf2': PPOTF2Policy}[args.framework]

    class EntropyPolicy(policy_cls):

        def loss_fn(policy, model, dist_class, train_batch):
            logits, _ = model(train_batch)
            action_dist = dist_class(logits, model)
            if args.framework == 'torch':
                model.tower_stats['policy_loss'] = torch.tensor([0.0])
                policy.policy_loss = torch.mean(-0.1 * action_dist.entropy() - action_dist.logp(train_batch['actions']) * train_batch['advantages'])
            else:
                policy.policy_loss = -0.1 * action_dist.entropy() - tf.reduce_mean(action_dist.logp(train_batch['actions']) * train_batch['advantages'])
            return policy.policy_loss

    class EntropyLossPPO(PPO):

        @classmethod
        def get_default_policy_class(cls, config):
            return EntropyPolicy
    run_heuristic_vs_learned(args, use_lstm=True, algorithm_config=PPOConfig(algo_class=EntropyLossPPO))