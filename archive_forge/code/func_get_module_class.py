from typing import Optional, Type, Union, TYPE_CHECKING
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.core.learner.learner import (
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.testing.testing_learner import BaseTestingLearnerHyperparameters
from ray.rllib.core.rl_module.marl_module import (
@DeveloperAPI
def get_module_class(framework: str) -> Type['RLModule']:
    if framework == 'tf2':
        from ray.rllib.core.testing.tf.bc_module import DiscreteBCTFModule
        return DiscreteBCTFModule
    elif framework == 'torch':
        from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule
        return DiscreteBCTorchModule
    else:
        raise ValueError(f'Unsupported framework: {framework}')