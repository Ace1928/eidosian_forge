from typing import Optional, Type, Union, TYPE_CHECKING
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.core.learner.learner import (
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.testing.testing_learner import BaseTestingLearnerHyperparameters
from ray.rllib.core.rl_module.marl_module import (
@DeveloperAPI
def get_optimizer_default_class(framework: str) -> Type[Optimizer]:
    if framework == 'tf2':
        import tensorflow as tf
        return tf.keras.optimizers.Adam
    elif framework == 'torch':
        import torch
        return torch.optim.Adam
    else:
        raise ValueError(f'Unsupported framework: {framework}')