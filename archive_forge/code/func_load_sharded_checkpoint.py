import collections
import copy
import functools
import gc
import importlib.metadata
import inspect
import itertools
import json
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from zipfile import is_zipfile
import torch
from packaging import version
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, Identity
from torch.utils.checkpoint import checkpoint
from .activations import get_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, GenerationMixin
from .integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled
from .pytorch_utils import (  # noqa: F401
from .quantizers import AutoHfQuantizer, HfQuantizer
from .safetensors_conversion import auto_conversion
from .utils import (
from .utils.hub import convert_file_size_to_int, create_and_tag_model_card, get_checkpoint_shard_files
from .utils.import_utils import (
from .utils.quantization_config import BitsAndBytesConfig, QuantizationMethod
def load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True):
    """
    This is the same as
    [`torch.nn.Module.load_state_dict`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict)
    but for a sharded checkpoint.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`torch.nn.Module`): The model in which to load the checkpoint.
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        strict (`bool`, *optional`, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
        prefer_safe (`bool`, *optional*, defaults to `False`)
            If both safetensors and PyTorch save files are present in checkpoint and `prefer_safe` is True, the
            safetensors files will be loaded. Otherwise, PyTorch files are always loaded when possible.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)
    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)
    if not index_present and (not (safe_index_present and is_safetensors_available())):
        filenames = (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME) if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")
    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():
                load_safe = True
            else:
                logger.warning(f'Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!')
        elif not index_present:
            load_safe = True
    load_index = safe_index_file if load_safe else index_file
    with open(load_index, 'r', encoding='utf-8') as f:
        index = json.load(f)
    shard_files = list(set(index['weight_map'].values()))
    loaded_keys = index['weight_map'].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f'Error(s) in loading state_dict for {model.__class__.__name__}'
        if len(missing_keys) > 0:
            str_missing_keys = ','.join([f'"{k}"' for k in missing_keys])
            error_message += f'\nMissing key(s): {str_missing_keys}.'
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ','.join([f'"{k}"' for k in unexpected_keys])
            error_message += f'\nMissing key(s): {str_unexpected_keys}.'
        raise RuntimeError(error_message)
    weights_only_kwarg = {'weights_only': True} if is_torch_greater_or_equal_than_1_13 else {}
    loader = safe_load_file if load_safe else partial(torch.load, map_location='cpu', **weights_only_kwarg)
    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        model.load_state_dict(state_dict, strict=False)
        del state_dict
        gc.collect()
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)