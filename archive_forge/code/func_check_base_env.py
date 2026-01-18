import logging
import traceback
from copy import copy
from typing import TYPE_CHECKING, Optional, Set, Union
import numpy as np
import tree  # pip install dm_tree
from ray.actor import ActorHandle
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.error import ERR_MSG_OLD_GYM_API, UnsupportedSpaceException
from ray.rllib.utils.gym import check_old_gym_env, try_import_gymnasium_and_gym
from ray.rllib.utils.spaces.space_utils import (
from ray.rllib.utils.typing import EnvType
from ray.util import log_once
@DeveloperAPI
def check_base_env(env: 'BaseEnv') -> None:
    """Checking for common errors in RLlib BaseEnvs.

    Args:
        env: The env to be checked.
    """
    from ray.rllib.env import BaseEnv
    if not isinstance(env, BaseEnv):
        raise ValueError('The passed env is not a BaseEnv.')
    try:
        obs_and_infos = env.try_reset(seed=42, options={})
        check_old_gym_env(reset_results=obs_and_infos)
    except Exception as e:
        raise ValueError(ERR_MSG_OLD_GYM_API.format(env, 'In particular, the `try_reset()` method seems to be faulty.')) from e
    reset_obs, reset_infos = obs_and_infos
    sampled_obs = env.observation_space_sample()
    _check_if_multi_env_dict(env, reset_obs, 'try_reset')
    _check_if_multi_env_dict(env, sampled_obs, 'observation_space_sample()')
    try:
        env.observation_space_contains(reset_obs)
    except Exception as e:
        raise ValueError('Your observation_space_contains function has some error ') from e
    if not env.observation_space_contains(reset_obs):
        error = _not_contained_error('try_reset', 'observation') + f': \n\n reset_obs: {reset_obs}\n\n env.observation_space_sample(): {sampled_obs}\n\n '
        raise ValueError(error)
    if not env.observation_space_contains(sampled_obs):
        error = _not_contained_error('observation_space_sample', 'observation') + f': \n\n sampled_obs: {sampled_obs}\n\n '
        raise ValueError(error)
    sampled_action = env.action_space_sample()
    try:
        env.action_space_contains(sampled_action)
    except Exception as e:
        raise ValueError('Your action_space_contains function has some error ') from e
    if not env.action_space_contains(sampled_action):
        error = _not_contained_error('action_space_sample', 'action') + f': \n\n sampled_action {sampled_action}\n\n'
        raise ValueError(error)
    _check_if_multi_env_dict(env, sampled_action, 'action_space_sample()')
    env.send_actions(sampled_action)
    try:
        results = env.poll()
        check_old_gym_env(step_results=results[:-1])
    except Exception as e:
        raise ValueError(ERR_MSG_OLD_GYM_API.format(env, 'In particular, the `poll()` method seems to be faulty.')) from e
    next_obs, reward, done, truncated, info, _ = results
    _check_if_multi_env_dict(env, next_obs, 'step, next_obs')
    _check_if_multi_env_dict(env, reward, 'step, reward')
    _check_if_multi_env_dict(env, done, 'step, done')
    _check_if_multi_env_dict(env, truncated, 'step, truncated')
    _check_if_multi_env_dict(env, info, 'step, info')
    if not env.observation_space_contains(next_obs):
        error = _not_contained_error('poll', 'observation') + f': \n\n reset_obs: {reset_obs}\n\n env.step():{next_obs}\n\n'
        raise ValueError(error)
    _check_reward(reward, base_env=True, agent_ids=env.get_agent_ids())
    _check_done_and_truncated(done, truncated, base_env=True, agent_ids=env.get_agent_ids())
    _check_info(info, base_env=True, agent_ids=env.get_agent_ids())