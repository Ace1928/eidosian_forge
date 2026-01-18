from collections import OrderedDict
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, MultiDiscrete
from gymnasium.wrappers import EnvCompatibility
import numpy as np
from recsim.document import AbstractDocumentSampler
from recsim.simulator import environment, recsim_gym
from recsim.user import AbstractUserModel, AbstractResponse
from typing import Callable, List, Optional, Type
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.gym import convert_old_gym_space_to_gymnasium_space
from ray.rllib.utils.spaces.space_utils import convert_element_to_space_type
def recsim_gym_wrapper(recsim_gym_env: gym.Env, convert_to_discrete_action_space: bool=False, wrap_for_bandits: bool=False) -> gym.Env:
    """Makes sure a RecSim gym.Env can ba handled by RLlib.

    In RecSim's observation spaces, the "doc" field is a dictionary keyed by
    document IDs. Those IDs are changing every step, thus generating a
    different observation space in each time. This causes issues for RLlib
    because it expects the observation space to remain the same across steps.

    Also, RecSim's reset() function returns an observation without the
    "response" field, breaking RLlib's check. This wrapper fixes that by
    assigning a random "response".

    Args:
        recsim_gym_env: The RecSim gym.Env instance. Usually resulting from a
            raw RecSim env having been passed through RecSim's utility function:
            `recsim.simulator.recsim_gym.RecSimGymEnv()`.
        convert_to_discrete_action_space: Optional bool indicating, whether
            the action space of the created env class should be Discrete
            (rather than MultiDiscrete, even if slate size > 1). This is useful
            for algorithms that don't support MultiDiscrete action spaces,
            such as RLlib's DQN. If None, `convert_to_discrete_action_space`
            may also be provided via the EnvContext (config) when creating an
            actual env instance.
        wrap_for_bandits: Bool indicating, whether this RecSim env should be
            wrapped for use with our Bandits agent.

    Returns:
        An RLlib-ready gym.Env instance.
    """
    env = RecSimResetWrapper(recsim_gym_env)
    env = RecSimObservationSpaceWrapper(env)
    if convert_to_discrete_action_space:
        env = MultiDiscreteToDiscreteActionWrapper(env)
    if wrap_for_bandits:
        env = RecSimObservationBanditWrapper(env)
    return env