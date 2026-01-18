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
def run_same_policy(args, stop):
    """Use the same policy for both agents (trivial case)."""
    config = PPOConfig().environment('RockPaperScissors').framework(args.framework)
    results = tune.Tuner('PPO', param_space=config, run_config=air.RunConfig(stop=stop, verbose=1)).fit()
    if args.as_test:
        check_learning_achieved(results, 0.0)