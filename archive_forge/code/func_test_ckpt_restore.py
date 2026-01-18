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
def test_ckpt_restore(config: 'AlgorithmConfig', env_name: str, tf2=False, replay_buffer=False, run_restored_algorithm=True, eval_workerset=False):
    """Test that after an algorithm is trained, its checkpoint can be restored.

    Check the replay buffers of the algorithm to see if they have identical data.
    Check the optimizer weights of the policy on the algorithm to see if they're
    identical.

    Args:
        config: The config of the algorithm to be trained.
        env_name: The name of the gymansium environment to be trained on.
        tf2: Whether to test the algorithm with the tf2 framework or not.
        object_store: Whether to test checkpointing with objects from the object store.
        replay_buffer: Whether to test checkpointing with replay buffers.
        run_restored_algorithm: Whether to run the restored algorithm after restoring.

    """
    if replay_buffer:
        config['store_buffer_in_checkpoints'] = True
    frameworks = (['tf2'] if tf2 else []) + ['torch', 'tf']
    for fw in framework_iterator(config, frameworks=frameworks):
        env = gym.make(env_name)
        alg1 = config.environment(env_name).framework(fw).build()
        alg2 = config.environment(env_name).build()
        policy1 = alg1.get_policy()
        res = alg1.train()
        print('current status: ' + str(res))
        optim_state = policy1.get_state().get('_optimizer_variables')
        checkpoint = alg1.save()
        for num_restores in range(2):
            alg2.restore(checkpoint)
        if optim_state:
            s2 = alg2.get_policy().get_state().get('_optimizer_variables')
            if fw in ['tf2', 'tf']:
                check(s2, optim_state)
            else:
                for i, s2_ in enumerate(s2):
                    check(list(s2_['state'].values()), list(optim_state[i]['state'].values()))
        if replay_buffer:
            data = alg1.local_replay_buffer.replay_buffers['default_policy']._storage[42:42 + 42]
            new_data = alg2.local_replay_buffer.replay_buffers['default_policy']._storage[42:42 + 42]
            check(data, new_data)
        if eval_workerset:
            eval_mapping_src = inspect.getsource(alg1.evaluation_workers.local_worker().policy_mapping_fn)
            check(eval_mapping_src, inspect.getsource(alg2.evaluation_workers.local_worker().policy_mapping_fn))
            check(eval_mapping_src, inspect.getsource(alg2.workers.local_worker().policy_mapping_fn), false=True)
        for _ in range(1):
            obs = env.observation_space.sample()
            a1 = _get_mean_action_from_algorithm(alg1, obs)
            a2 = _get_mean_action_from_algorithm(alg2, obs)
            print('Checking computed actions', alg1, obs, a1, a2)
            if abs(a1 - a2) > 0.1:
                raise AssertionError('algo={} [a1={} a2={}]'.format(str(alg1.__class__), a1, a2))
        alg1.stop()
        if run_restored_algorithm:
            print('Starting second run on Algo 2...')
            alg2.train()
        alg2.stop()