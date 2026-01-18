from typing import Optional, Type, Union, TYPE_CHECKING
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.core.learner.learner import (
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.testing.testing_learner import BaseTestingLearnerHyperparameters
from ray.rllib.core.rl_module.marl_module import (
@DeveloperAPI
def get_module_spec(framework: str, env: 'gym.Env', is_multi_agent: bool=False):
    spec = SingleAgentRLModuleSpec(module_class=get_module_class(framework), observation_space=env.observation_space, action_space=env.action_space, model_config_dict={'fcnet_hiddens': [32]})
    if is_multi_agent:
        return MultiAgentRLModuleSpec(marl_module_class=MultiAgentRLModule, module_specs={DEFAULT_POLICY_ID: spec})
    else:
        return spec