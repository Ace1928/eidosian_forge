from typing import Optional, Type, TYPE_CHECKING, Union
from ray.rllib.core.learner.learner import (
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.from_config import NotProvided
def framework(self, eager_tracing: Optional[bool]=NotProvided, torch_compile: Optional[bool]=NotProvided, torch_compile_cfg: Optional['TorchCompileConfig']=NotProvided, torch_compile_what_to_compile: Optional[str]=NotProvided) -> 'LearnerGroupConfig':
    if eager_tracing is not NotProvided:
        self.eager_tracing = eager_tracing
    if torch_compile is not NotProvided:
        self.torch_compile = torch_compile
    if torch_compile_cfg is not NotProvided:
        self.torch_compile_cfg = torch_compile_cfg
    if torch_compile_what_to_compile is not NotProvided:
        self.torch_compile_what_to_compile = torch_compile_what_to_compile
    return self