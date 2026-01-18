import inspect
from copy import deepcopy
import numpy as np
import gym
from gym import logger, spaces
from gym.utils.passive_env_checker import (
def check_reset_options(env: gym.Env):
    """Check that the environment can be reset with options.

    Args:
        env: The environment to check

    Raises:
        AssertionError: The environment cannot be reset with options,
            even though `options` or `kwargs` appear in the signature.
    """
    signature = inspect.signature(env.reset)
    if 'options' in signature.parameters or ('kwargs' in signature.parameters and signature.parameters['kwargs'].kind is inspect.Parameter.VAR_KEYWORD):
        try:
            env.reset(options={})
        except TypeError as e:
            raise AssertionError(f'The environment cannot be reset with options, even though `options` or `**kwargs` appear in the signature. This should never happen, please report this issue. The error was: {e}')
    else:
        raise gym.error.Error('The `reset` method does not provide an `options` or `**kwargs` keyword argument.')