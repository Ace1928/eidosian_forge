import functools
from typing import List
import torch
import torch.distributed._shard.sharding_spec as shard_spec
from .api import (
from .metadata import ShardMetadata  # noqa: F401
from torch.distributed._shard.op_registry_utils import _decorator_func
from ._ops import *  # noqa: F403
def pre_load_state_dict_hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    """
    Pre-load state dict hook to add ShardedTensor to the module.
    """
    for submodule_name, submodule in module.named_modules():
        for attr_name in submodule.__dict__.keys():
            mod_prefix = prefix + submodule_name
            key = mod_prefix + ('.' if mod_prefix else '') + attr_name
            if key in state_dict:
                if isinstance(state_dict[key], ShardedTensor):
                    setattr(submodule, attr_name, state_dict[key])