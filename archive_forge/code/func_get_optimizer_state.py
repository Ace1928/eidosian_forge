import logging
import pathlib
from typing import (
from ray.rllib.core.learner.learner import (
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModule
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchDDPRLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import (
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.torch_utils import (
from ray.rllib.utils.typing import Optimizer, Param, ParamDict, TensorType
@override(Learner)
def get_optimizer_state(self) -> Mapping[str, Any]:
    optimizer_name_state = {}
    for name, optim in self._named_optimizers.items():
        optim_state_dict = optim.state_dict()
        optim_state_dict_cpu = copy_torch_tensors(optim_state_dict, device='cpu')
        optimizer_name_state[name] = optim_state_dict_cpu
    return optimizer_name_state