from typing import Any, Mapping
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.rl_module.marl_module import (
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.nested_dict import NestedDict
def get_train_action_dist_cls(self):
    return TorchCategorical