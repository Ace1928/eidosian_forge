import collections
import os
import platform
import re
import socket
from contextlib import contextmanager
from functools import partial, reduce
from types import MethodType
from typing import OrderedDict
import torch
from packaging.version import Version
from safetensors.torch import save_file as safe_save_file
from ..commands.config.default import write_basic_config  # noqa: F401
from ..logging import get_logger
from ..state import PartialState
from .constants import FSDP_PYTORCH_VERSION
from .dataclasses import DistributedType
from .imports import is_deepspeed_available, is_torch_distributed_available, is_torch_xla_available
from .modeling import id_tensor_storage
from .transformer_engine import convert_model
from .versions import is_torch_version
def clean_state_dict_for_safetensors(state_dict: dict):
    """
    Cleans the state dictionary from a model and removes tensor aliasing if present.

    Args:
        state_dict (`dict`):
            The state dictionary from a model
    """
    ptrs = collections.defaultdict(list)
    for name, tensor in state_dict.items():
        if not isinstance(tensor, str):
            ptrs[id_tensor_storage(tensor)].append(name)
    shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
    warn_names = set()
    for names in shared_ptrs.values():
        found_names = [name for name in names if name in state_dict]
        warn_names.update(found_names[1:])
        for name in found_names[1:]:
            del state_dict[name]
    if len(warn_names) > 0:
        logger.warning(f"Removed shared tensor {warn_names} while saving. This should be OK, but check by verifying that you don't receive any warning while reloading")
    state_dict = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    return state_dict