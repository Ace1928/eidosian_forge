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
def check_vector_env(env: 'VectorEnv') -> None:
    """Checking for common errors in RLlib VectorEnvs.

    Args:
        env: The env to be checked.
    """
    sampled_obs = env.observation_space.sample()
    try:
        vector_reset = env.vector_reset(seeds=[42] * env.num_envs, options=[{}] * env.num_envs)
    except Exception as e:
        raise ValueError("Your Env's `vector_reset()` method has some error! Make sure it expects a list of `seeds` (int) as well as a list of `options` dicts as optional, named args, e.g. def vector_reset(self, index: int, *, seeds: Optional[List[int]] = None, options: Optional[List[dict]] = None)") from e
    if not isinstance(vector_reset, tuple) or len(vector_reset) != 2:
        raise ValueError(f'The `vector_reset()` method of your env must return a Tuple[obs, infos] as of gym>=0.26! Your method returned: {vector_reset}.')
    reset_obs, reset_infos = vector_reset
    if not isinstance(reset_obs, list) or len(reset_obs) != env.num_envs:
        raise ValueError(f"The observations returned by your env's `vector_reset()` method is NOT a list or do not contain exactly `num_envs` ({env.num_envs}) items! Your observations were: {reset_obs}")
    if not isinstance(reset_infos, list) or len(reset_infos) != env.num_envs:
        raise ValueError(f"The infos returned by your env's `vector_reset()` method is NOT a list or do not contain exactly `num_envs` ({env.num_envs}) items! Your infos were: {reset_infos}")
    try:
        env.observation_space.contains(reset_obs[0])
    except Exception as e:
        raise ValueError('Your `observation_space.contains` function has some error!') from e
    if not env.observation_space.contains(reset_obs[0]):
        error = _not_contained_error('vector_reset', 'observation') + f': \n\n reset_obs: {reset_obs}\n\n env.observation_space.sample(): {sampled_obs}\n\n '
        raise ValueError(error)
    try:
        reset_at = env.reset_at(index=0, seed=42, options={})
    except Exception as e:
        raise ValueError("Your Env's `reset_at()` method has some error! Make sure it expects a vector index (int) and an optional seed (int) as args.") from e
    if not isinstance(reset_at, tuple) or len(reset_at) != 2:
        raise ValueError(f'The `reset_at()` method of your env must return a Tuple[obs, infos] as of gym>=0.26! Your method returned: {reset_at}.')
    reset_obs, reset_infos = reset_at
    if not isinstance(reset_infos, dict):
        raise ValueError(f'The `reset_at()` method of your env must return an info dict as second return value! Your method returned {reset_infos}')
    if not env.observation_space.contains(reset_obs):
        error = _not_contained_error('try_reset', 'observation') + f': \n\n reset_obs: {reset_obs}\n\n env.observation_space.sample(): {sampled_obs}\n\n '
        raise ValueError(error)
    if not env.observation_space.contains(sampled_obs):
        error = _not_contained_error('observation_space.sample()', 'observation') + f': \n\n sampled_obs: {sampled_obs}\n\n '
        raise ValueError(error)
    sampled_action = env.action_space.sample()
    if not env.action_space.contains(sampled_action):
        error = _not_contained_error('action_space.sample()', 'action') + f': \n\n sampled_action {sampled_action}\n\n'
        raise ValueError(error)
    step_results = env.vector_step([sampled_action for _ in range(env.num_envs)])
    if not isinstance(step_results, tuple) or len(step_results) != 5:
        raise ValueError(f'The `vector_step()` method of your env must return a Tuple[List[obs], List[rewards], List[terminateds], List[truncateds], List[infos]] as of gym>=0.26! Your method returned: {step_results}.')
    obs, rewards, terminateds, truncateds, infos = step_results
    _check_if_vetor_env_list(env, obs, 'step, obs')
    _check_if_vetor_env_list(env, rewards, 'step, rewards')
    _check_if_vetor_env_list(env, terminateds, 'step, terminateds')
    _check_if_vetor_env_list(env, truncateds, 'step, truncateds')
    _check_if_vetor_env_list(env, infos, 'step, infos')
    if not env.observation_space.contains(obs[0]):
        error = _not_contained_error('vector_step', 'observation') + f': \n\n obs: {obs[0]}\n\n env.vector_step():{obs}\n\n'
        raise ValueError(error)
    _check_reward(rewards[0], base_env=False)
    _check_done_and_truncated(terminateds[0], truncateds[0], base_env=False)
    _check_info(infos[0], base_env=False)