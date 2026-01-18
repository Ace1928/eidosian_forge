from typing import Iterable, List, Optional, Union
import gym
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.vector.sync_vector_env import SyncVectorEnv
from gym.vector.vector_env import VectorEnv, VectorEnvWrapper
def create_env(env_num: int):
    """Creates an environment that can enable or disable the environment checker."""
    _disable_env_checker = True if env_num > 0 else disable_env_checker

    def _make_env():
        env = gym.envs.registration.make(id, disable_env_checker=_disable_env_checker, **kwargs)
        if wrappers is not None:
            if callable(wrappers):
                env = wrappers(env)
            elif isinstance(wrappers, Iterable) and all([callable(w) for w in wrappers]):
                for wrapper in wrappers:
                    env = wrapper(env)
            else:
                raise NotImplementedError
        return env
    return _make_env