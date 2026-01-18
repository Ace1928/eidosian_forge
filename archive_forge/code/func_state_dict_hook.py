import functools
from typing import List
import torch
import torch.distributed._shard.sharding_spec as shard_spec
from .api import (
from .metadata import ShardMetadata  # noqa: F401
from torch.distributed._shard.op_registry_utils import _decorator_func
from ._ops import *  # noqa: F403
def state_dict_hook(module, destination, prefix, local_metadata):
    """
    Hook to add ShardedTensor to Module's ``state_dict``. Needs to be
    registered to the Module using
    :meth:`torch.nn.Module._register_state_dict_hook`.
    """
    for submodule_name, submodule in module.named_modules():
        for attr_name, attr in submodule.__dict__.items():
            if isinstance(attr, ShardedTensor):
                mod_prefix = prefix + submodule_name
                key = mod_prefix + ('.' if mod_prefix else '') + attr_name
                destination[key] = attr