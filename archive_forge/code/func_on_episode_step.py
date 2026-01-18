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
def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: int, **kwargs):
    assert episode.length > 0, 'ERROR: `on_episode_step()` callback should not be called right after env reset!'
    pole_angle = abs(episode.last_observation_for()[2])
    raw_angle = abs(episode.last_raw_obs_for()[2])
    assert pole_angle == raw_angle
    episode.user_data['pole_angles'].append(pole_angle)
    if np.abs(episode.last_info_for()['pole_angle_vel']) > 0.25:
        print('This is a fast pole!')