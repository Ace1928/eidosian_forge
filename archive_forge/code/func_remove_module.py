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
def remove_module(self, module_id: ModuleID) -> None:
    super().remove_module(module_id)
    if self._torch_compile_complete_update:
        torch._dynamo.reset()
        self._compiled_update_initialized = False
        torch_compile_cfg = self._framework_hyperparameters.torch_compile_cfg
        self._possibly_compiled_update = torch.compile(self._uncompiled_update, backend=torch_compile_cfg.torch_dynamo_backend, mode=torch_compile_cfg.torch_dynamo_mode, **torch_compile_cfg.kwargs)