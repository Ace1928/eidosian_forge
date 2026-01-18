import logging
from gymnasium.spaces import Discrete
import numpy as np
from ray.rllib.utils.annotations import override
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.evaluation.rollout_worker import get_global_worker
from ray.rllib.env.base_env import BaseEnv, convert_to_base_env
from ray.rllib.utils.typing import EnvType
def model_vector_env(env: EnvType) -> BaseEnv:
    """Returns a VectorizedEnv wrapper around the given environment.

    To obtain worker configs, one can call get_global_worker().

    Args:
        env: The input environment (of any supported environment
            type) to be convert to a _VectorizedModelGymEnv (wrapped as
            an RLlib BaseEnv).

    Returns:
        BaseEnv: The BaseEnv converted input `env`.
    """
    worker = get_global_worker()
    worker_index = worker.worker_index
    if worker_index:
        env = _VectorizedModelGymEnv(make_env=worker.make_sub_env_fn, existing_envs=[env], num_envs=worker.config.num_envs_per_worker, observation_space=env.observation_space, action_space=env.action_space)
    return convert_to_base_env(env, make_env=worker.make_sub_env_fn, num_envs=worker.config.num_envs_per_worker, remote_envs=False, remote_env_batch_wait_ms=0)