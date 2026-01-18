import atexit
import logging
from functools import partial
from types import FunctionType
from typing import Callable, Optional, Type, Union
import ray
import ray.cloudpickle as pickle
from ray.experimental.internal_kv import (
from ray.tune.error import TuneError
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def register_env(name: str, env_creator: Callable):
    """Register a custom environment for use with RLlib.

    This enables the environment to be accessed on every Ray process
    in the cluster.

    Args:
        name: Name to register.
        env_creator: Callable that creates an env.
    """
    if not callable(env_creator):
        raise TypeError('Second argument must be callable.', env_creator)
    _global_registry.register(ENV_CREATOR, name, env_creator)