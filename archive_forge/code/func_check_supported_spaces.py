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
def check_supported_spaces(alg: str, config: 'AlgorithmConfig', train: bool=True, check_bounds: bool=False, frameworks: Optional[Tuple[str]]=None, use_gpu: bool=False):
    """Checks whether the given algorithm supports different action and obs spaces.

        Performs the checks by constructing an rllib algorithm from the config and
        checking to see that the model inside the policy is the correct one given
        the action and obs spaces. For example if the action space is discrete and
        the obs space is an image, then the model should be a vision network with
        a categorical action distribution.

    Args:
        alg: The name of the algorithm to test.
        config: The config to use for the algorithm.
        train: Whether to train the algorithm for a few iterations.
        check_bounds: Whether to check the bounds of the action space.
        frameworks: The frameworks to test the algorithm with.
        use_gpu: Whether to check support for training on a gpu.


    """
    from ray.rllib.examples.env.random_env import RandomEnv
    from ray.rllib.models.tf.complex_input_net import ComplexInputNetwork as ComplexNet
    from ray.rllib.models.tf.fcnet import FullyConnectedNetwork as FCNet
    from ray.rllib.models.tf.visionnet import VisionNetwork as VisionNet
    from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork as TorchComplexNet
    from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
    from ray.rllib.models.torch.visionnet import VisionNetwork as TorchVisionNet
    action_spaces_to_test = {'discrete': Discrete(5), 'continuous': Box(-1.0, 1.0, (5,), dtype=np.float32), 'int_actions': Box(0, 3, (2, 3), dtype=np.int32), 'multidiscrete': MultiDiscrete([1, 2, 3, 4]), 'tuple': GymTuple([Discrete(2), Discrete(3), Box(-1.0, 1.0, (5,), dtype=np.float32)]), 'dict': GymDict({'action_choice': Discrete(3), 'parameters': Box(-1.0, 1.0, (1,), dtype=np.float32), 'yet_another_nested_dict': GymDict({'a': GymTuple([Discrete(2), Discrete(3)])})})}
    observation_spaces_to_test = {'multi_binary': MultiBinary([3, 10, 10]), 'discrete': Discrete(5), 'continuous': Box(-1.0, 1.0, (5,), dtype=np.float32), 'vector2d': Box(-1.0, 1.0, (5, 5), dtype=np.float32), 'image': Box(-1.0, 1.0, (84, 84, 1), dtype=np.float32), 'vizdoomgym': Box(-1.0, 1.0, (240, 320, 3), dtype=np.float32), 'tuple': GymTuple([Discrete(10), Box(-1.0, 1.0, (5,), dtype=np.float32)]), 'dict': GymDict({'task': Discrete(10), 'position': Box(-1.0, 1.0, (5,), dtype=np.float32)})}
    rlmodule_supported_observation_spaces = ['multi_binary', 'discrete', 'continuous', 'image', 'vizdoomgym', 'tuple', 'dict']
    rlmodule_supported_frameworks = ('torch', 'tf2')
    rlmodule_supported_action_spaces = ['discrete', 'continuous']
    default_observation_space = default_action_space = 'discrete'
    config['log_level'] = 'ERROR'
    config['env'] = RandomEnv

    def _do_check(alg, config, a_name, o_name):
        config_copy = config.copy()
        config_copy.validate()
        if config_copy._enable_new_api_stack:
            if o_name not in rlmodule_supported_observation_spaces:
                logger.warning('Skipping PPO test with RLModules for obs space {}'.format(o_name))
                return
            if a_name not in rlmodule_supported_action_spaces:
                logger.warning('Skipping PPO test with RLModules for action space {}'.format(a_name))
                return
        fw = config['framework']
        action_space = action_spaces_to_test[a_name]
        obs_space = observation_spaces_to_test[o_name]
        print('=== Testing {} (fw={}) action_space={} obs_space={} ==='.format(alg, fw, action_space, obs_space))
        t0 = time.time()
        config.update_from_dict(dict(env_config=dict(action_space=action_space, observation_space=obs_space, reward_space=Box(1.0, 1.0, shape=(), dtype=np.float32), p_terminated=1.0, check_action_bounds=check_bounds)))
        stat = 'ok'
        try:
            algo = config.build()
        except ray.exceptions.RayActorError as e:
            if len(e.args) >= 2 and isinstance(e.args[2], UnsupportedSpaceException):
                stat = 'unsupported'
            elif isinstance(e.args[0].args[2], UnsupportedSpaceException):
                stat = 'unsupported'
            else:
                raise
        except UnsupportedSpaceException:
            stat = 'unsupported'
        else:
            if alg not in ['SAC', 'PPO']:
                if o_name in ['atari', 'image']:
                    if fw == 'torch':
                        assert isinstance(algo.get_policy().model, TorchVisionNet)
                    else:
                        assert isinstance(algo.get_policy().model, VisionNet)
                elif o_name == 'continuous':
                    if fw == 'torch':
                        assert isinstance(algo.get_policy().model, TorchFCNet)
                    else:
                        assert isinstance(algo.get_policy().model, FCNet)
                elif o_name == 'vector2d':
                    if fw == 'torch':
                        assert isinstance(algo.get_policy().model, (TorchComplexNet, TorchFCNet))
                    else:
                        assert isinstance(algo.get_policy().model, (ComplexNet, FCNet))
            if train:
                algo.train()
            algo.stop()
        print('Test: {}, ran in {}s'.format(stat, time.time() - t0))
    if not frameworks:
        frameworks = ('tf2', 'tf', 'torch')
    if config._enable_new_api_stack:
        frameworks = tuple((fw for fw in frameworks if fw in rlmodule_supported_frameworks))
    _do_check_remote = ray.remote(_do_check)
    _do_check_remote = _do_check_remote.options(num_gpus=1 if use_gpu else 0)
    for _ in framework_iterator(config, frameworks=frameworks):
        for a_name in action_spaces_to_test.keys():
            o_name = default_observation_space
            ray.get(_do_check_remote.remote(alg, config, a_name, o_name))
        for o_name in observation_spaces_to_test.keys():
            a_name = default_action_space
            ray.get(_do_check_remote.remote(alg, config, a_name, o_name))