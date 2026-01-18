import logging
import os
import shutil
import tempfile
from typing import Any, Dict
import torch
from packaging.version import Version
import ray
from ray import train
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train import Checkpoint
from ray.util import PublicAPI
def lightning_module_state_dict(self) -> Dict[str, Any]:
    """Gathers the full state dict to rank 0 on CPU."""
    assert self.model is not None, 'Failed to get the state dict for a None model!'
    if _LIGHTNING_GREATER_EQUAL_2_0 and _TORCH_FSDP_AVAILABLE:
        with FullyShardedDataParallel.state_dict_type(module=self.model, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state_dict = self.model.state_dict()
            prefix_len = len('_forward_module.')
            return {k[prefix_len:]: v for k, v in state_dict.items()}
    else:
        return super().lightning_module_state_dict()