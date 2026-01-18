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
def check_inference_w_connectors(policy, env_name, max_steps: int=100):
    """Checks whether the given policy can infer actions from an env with connectors.

    Args:
        policy: The policy to check.
        env_name: Name of the environment to check
        max_steps: The maximum number of steps to run the environment for.

    Raises:
        ValueError: If the policy cannot infer actions from the environment.
    """
    from ray.rllib.utils.policy import local_policy_inference
    env = gym.make(env_name)
    if is_atari(env):
        env = wrap_deepmind(env, dim=policy.config['model']['dim'], framestack=policy.config['model'].get('framestack'))
    obs, info = env.reset()
    reward, terminated, truncated = (0.0, False, False)
    ts = 0
    while not terminated and (not truncated) and (ts < max_steps):
        action_out = local_policy_inference(policy, env_id=0, agent_id=0, obs=obs, reward=reward, terminated=terminated, truncated=truncated, info=info)
        obs, reward, terminated, truncated, info = env.step(action_out[0][0])
        ts += 1