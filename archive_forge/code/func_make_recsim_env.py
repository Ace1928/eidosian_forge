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
def make_recsim_env(recsim_user_model_creator: Callable[[EnvContext], AbstractUserModel], recsim_document_sampler_creator: Callable[[EnvContext], AbstractDocumentSampler], reward_aggregator: Callable[[List[AbstractResponse]], float]) -> Type[gym.Env]:
    """Creates a RLlib-ready gym.Env class given RecSim user and doc models.

    See https://github.com/google-research/recsim for more information on how to
    build the required components from scratch in python using RecSim.

    Args:
        recsim_user_model_creator: A callable taking an EnvContext and returning
            a RecSim AbstractUserModel instance to use.
        recsim_document_sampler_creator: A callable taking an EnvContext and
            returning a RecSim AbstractDocumentSampler
            to use. This will include a AbstractDocument as well.
        reward_aggregator: Callable taking a list of RecSim
            AbstractResponse instances and returning a float (aggregated
            reward).

    Returns:
        An RLlib-ready gym.Env class to use inside an Algorithm.
    """

    class _RecSimEnv(gym.Wrapper):

        def __init__(self, config: Optional[EnvContext]=None):
            default_config = {'num_candidates': 10, 'slate_size': 2, 'resample_documents': True, 'seed': 0, 'convert_to_discrete_action_space': False, 'wrap_for_bandits': False}
            if config is None or isinstance(config, dict):
                config = EnvContext(config or default_config, worker_index=0)
            config.set_defaults(default_config)
            recsim_user_model = recsim_user_model_creator(config)
            recsim_document_sampler = recsim_document_sampler_creator(config)
            raw_recsim_env = environment.SingleUserEnvironment(recsim_user_model, recsim_document_sampler, config['num_candidates'], config['slate_size'], resample_documents=config['resample_documents'])
            gym_env = recsim_gym.RecSimGymEnv(raw_recsim_env, reward_aggregator)
            gym_env = EnvCompatibility(gym_env)
            env = recsim_gym_wrapper(gym_env, config['convert_to_discrete_action_space'], config['wrap_for_bandits'])
            super().__init__(env=env)
    return _RecSimEnv