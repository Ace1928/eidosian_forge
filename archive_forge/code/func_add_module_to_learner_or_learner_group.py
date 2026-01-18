from typing import Optional, Type, Union, TYPE_CHECKING
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.core.learner.learner import (
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.testing.testing_learner import BaseTestingLearnerHyperparameters
from ray.rllib.core.rl_module.marl_module import (
@DeveloperAPI
def add_module_to_learner_or_learner_group(framework: str, env: 'gym.Env', module_id: str, learner_group_or_learner: Union[LearnerGroup, 'Learner']):
    learner_group_or_learner.add_module(module_id=module_id, module_spec=get_module_spec(framework, env, is_multi_agent=False))