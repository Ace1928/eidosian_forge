from typing import Dict, Tuple
import argparse
import gymnasium as gym
import numpy as np
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs) -> None:
    result['sum_actions_in_train_batch'] = train_batch['actions'].sum()
    print('policy.learn_on_batch() result: {} -> sum actions: {}'.format(policy, result['sum_actions_in_train_batch']))