from collections import Counter
import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Tuple as GymTuple
import inspect
import logging
import numpy as np
import os
import pprint
import random
import re
import time
import tree  # pip install dm_tree
from typing import (
import yaml
import ray
from ray import air, tune
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.tune import CLIReporter, run_experiments
def check_train_results(train_results: ResultDict):
    """Checks proper structure of a Algorithm.train() returned dict.

    Args:
        train_results: The train results dict to check.

    Raises:
        AssertionError: If `train_results` doesn't have the proper structure or
            data in it.
    """
    from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
    from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LEARNER_STATS_KEY
    for key in ['agent_timesteps_total', 'config', 'custom_metrics', 'episode_len_mean', 'episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'hist_stats', 'info', 'iterations_since_restore', 'num_healthy_workers', 'perf', 'policy_reward_max', 'policy_reward_mean', 'policy_reward_min', 'sampler_perf', 'time_since_restore', 'time_this_iter_s', 'timesteps_total', 'timers', 'time_total_s', 'training_iteration']:
        assert key in train_results, f"'{key}' not found in `train_results` ({train_results})!"
    assert isinstance(train_results['config'], dict), '`config` in results not a python dict!'
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    is_multi_agent = AlgorithmConfig().update_from_dict({'policies': train_results['config']['policies']}).is_multi_agent()
    info = train_results['info']
    assert LEARNER_INFO in info, f"'learner' not in train_results['infos'] ({info})!"
    assert 'num_steps_trained' in info or NUM_ENV_STEPS_TRAINED in info, f"'num_(env_)?steps_trained' not in train_results['infos'] ({info})!"
    learner_info = info[LEARNER_INFO]
    if not is_multi_agent:
        assert len(learner_info) == 0 or DEFAULT_POLICY_ID in learner_info, f"'{DEFAULT_POLICY_ID}' not found in train_results['infos']['learner'] ({learner_info})!"
    for pid, policy_stats in learner_info.items():
        if pid == 'batch_count':
            continue
        if pid == '__all__':
            continue
        if LEARNER_STATS_KEY in policy_stats:
            learner_stats = policy_stats[LEARNER_STATS_KEY]
        else:
            learner_stats = policy_stats
        for key, value in learner_stats.items():
            if key.startswith('min_') or key.startswith('max_'):
                assert np.isscalar(value), f"'key' value not a scalar ({value})!"
    return train_results