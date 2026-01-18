import argparse
import gymnasium as gym
import os
import ray
from ray.rllib.algorithms.dqn import DQNConfig, DQNTFPolicy, DQNTorchPolicy
from ray.rllib.algorithms.ppo import (
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
def select_policy(algorithm, framework):
    if algorithm == 'PPO':
        if framework == 'torch':
            return PPOTorchPolicy
        elif framework == 'tf':
            return PPOTF1Policy
        else:
            return PPOTF2Policy
    elif algorithm == 'DQN':
        if framework == 'torch':
            return DQNTorchPolicy
        else:
            return DQNTFPolicy
    else:
        raise ValueError('Unknown algorithm: ', algorithm)