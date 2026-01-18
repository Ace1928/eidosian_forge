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
def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
    assert samples.count == 2000, f'I was expecting 2000 here, but got {samples.count}!'