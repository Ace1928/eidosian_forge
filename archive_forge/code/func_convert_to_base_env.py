import logging
from typing import Callable, Tuple, Optional, List, Dict, Any, TYPE_CHECKING, Union, Set
import gymnasium as gym
import ray
from ray.rllib.utils.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.rllib.utils.typing import AgentID, EnvID, EnvType, MultiEnvDict
@DeveloperAPI
def convert_to_base_env(env: EnvType, make_env: Callable[[int], EnvType]=None, num_envs: int=1, remote_envs: bool=False, remote_env_batch_wait_ms: int=0, worker: Optional['RolloutWorker']=None, restart_failed_sub_environments: bool=False) -> 'BaseEnv':
    """Converts an RLlib-supported env into a BaseEnv object.

    Supported types for the `env` arg are gym.Env, BaseEnv,
    VectorEnv, MultiAgentEnv, ExternalEnv, or ExternalMultiAgentEnv.

    The resulting BaseEnv is always vectorized (contains n
    sub-environments) to support batched forward passes, where n may also
    be 1. BaseEnv also supports async execution via the `poll` and
    `send_actions` methods and thus supports external simulators.

    TODO: Support gym3 environments, which are already vectorized.

    Args:
        env: An already existing environment of any supported env type
            to convert/wrap into a BaseEnv. Supported types are gym.Env,
            BaseEnv, VectorEnv, MultiAgentEnv, ExternalEnv, and
            ExternalMultiAgentEnv.
        make_env: A callable taking an int as input (which indicates the
            number of individual sub-environments within the final
            vectorized BaseEnv) and returning one individual
            sub-environment.
        num_envs: The number of sub-environments to create in the
            resulting (vectorized) BaseEnv. The already existing `env`
            will be one of the `num_envs`.
        remote_envs: Whether each sub-env should be a @ray.remote actor.
            You can set this behavior in your config via the
            `remote_worker_envs=True` option.
        remote_env_batch_wait_ms: The wait time (in ms) to poll remote
            sub-environments for, if applicable. Only used if
            `remote_envs` is True.
        worker: An optional RolloutWorker that owns the env. This is only
            used if `remote_worker_envs` is True in your config and the
            `on_sub_environment_created` custom callback needs to be called
            on each created actor.
        restart_failed_sub_environments: If True and any sub-environment (within
            a vectorized env) throws any error during env stepping, the
            Sampler will try to restart the faulty sub-environment. This is done
            without disturbing the other (still intact) sub-environment and without
            the RolloutWorker crashing.

    Returns:
        The resulting BaseEnv object.
    """
    from ray.rllib.env.remote_base_env import RemoteBaseEnv
    from ray.rllib.env.external_env import ExternalEnv
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    from ray.rllib.env.vector_env import VectorEnv, VectorEnvWrapper
    if remote_envs and num_envs == 1:
        raise ValueError('Remote envs only make sense to use if num_envs > 1 (i.e. environment vectorization is enabled).')
    if isinstance(env, (BaseEnv, MultiAgentEnv, VectorEnv, ExternalEnv)):
        return env.to_base_env(make_env=make_env, num_envs=num_envs, remote_envs=remote_envs, remote_env_batch_wait_ms=remote_env_batch_wait_ms, restart_failed_sub_environments=restart_failed_sub_environments)
    elif remote_envs:
        multiagent = ray.get(env._is_multi_agent.remote()) if hasattr(env, '_is_multi_agent') else False
        env = RemoteBaseEnv(make_env, num_envs, multiagent=multiagent, remote_env_batch_wait_ms=remote_env_batch_wait_ms, existing_envs=[env], worker=worker, restart_failed_sub_environments=restart_failed_sub_environments)
    else:
        env = VectorEnv.vectorize_gym_envs(make_env=make_env, existing_envs=[env], num_envs=num_envs, action_space=env.action_space, observation_space=env.observation_space, restart_failed_sub_environments=restart_failed_sub_environments)
        env = VectorEnvWrapper(env)
    assert isinstance(env, BaseEnv), env
    return env