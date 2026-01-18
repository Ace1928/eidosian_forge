import inspect
from copy import deepcopy
import numpy as np
import gym
from gym import logger, spaces
from gym.utils.passive_env_checker import (
def check_reset_seed(env: gym.Env):
    """Check that the environment can be reset with a seed.

    Args:
        env: The environment to check

    Raises:
        AssertionError: The environment cannot be reset with a random seed,
            even though `seed` or `kwargs` appear in the signature.
    """
    signature = inspect.signature(env.reset)
    if 'seed' in signature.parameters or ('kwargs' in signature.parameters and signature.parameters['kwargs'].kind is inspect.Parameter.VAR_KEYWORD):
        try:
            obs_1, info = env.reset(seed=123)
            assert obs_1 in env.observation_space, 'The observation returned by `env.reset(seed=123)` is not within the observation space.'
            assert env.unwrapped._np_random is not None, 'Expects the random number generator to have been generated given a seed was passed to reset. Mostly likely the environment reset function does not call `super().reset(seed=seed)`.'
            seed_123_rng = deepcopy(env.unwrapped._np_random)
            obs_2, info = env.reset(seed=123)
            assert obs_2 in env.observation_space, 'The observation returned by `env.reset(seed=123)` is not within the observation space.'
            if env.spec is not None and env.spec.nondeterministic is False:
                assert data_equivalence(obs_1, obs_2), 'Using `env.reset(seed=123)` is non-deterministic as the observations are not equivalent.'
            assert env.unwrapped._np_random.bit_generator.state == seed_123_rng.bit_generator.state, 'Mostly likely the environment reset function does not call `super().reset(seed=seed)` as the random generates are not same when the same seeds are passed to `env.reset`.'
            obs_3, info = env.reset(seed=456)
            assert obs_3 in env.observation_space, 'The observation returned by `env.reset(seed=456)` is not within the observation space.'
            assert env.unwrapped._np_random.bit_generator.state != seed_123_rng.bit_generator.state, 'Mostly likely the environment reset function does not call `super().reset(seed=seed)` as the random number generators are not different when different seeds are passed to `env.reset`.'
        except TypeError as e:
            raise AssertionError(f'The environment cannot be reset with a random seed, even though `seed` or `kwargs` appear in the signature. This should never happen, please report this issue. The error was: {e}')
        seed_param = signature.parameters.get('seed')
        if seed_param is not None and seed_param.default is not None:
            logger.warn(f'The default seed argument in reset should be `None`, otherwise the environment will by default always be deterministic. Actual default: {seed_param.default}')
    else:
        raise gym.error.Error('The `reset` method does not provide a `seed` or `**kwargs` keyword argument.')