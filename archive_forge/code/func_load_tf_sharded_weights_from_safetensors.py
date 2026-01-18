from __future__ import annotations
import functools
import gc
import inspect
import json
import os
import pickle
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import h5py
import numpy as np
import tensorflow as tf
from packaging.version import parse
from . import DataCollatorWithPadding, DefaultDataCollator
from .activations_tf import get_tf_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, TFGenerationMixin
from .tf_utils import (
from .utils import (
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
def load_tf_sharded_weights_from_safetensors(model, shard_files, ignore_mismatched_sizes=False, strict=False, _prefix=None):
    """
    This is the same as `load_tf_weights_from_safetensors` but for a sharded TF-format safetensors checkpoint.
    Detect missing and unexpected layers and load the TF weights from the shard file accordingly to their names and
    shapes.

    This load is performed efficiently: each checkpoint shard is loaded one by one in RAM and deleted after being
    loaded in the model.

    Args:
        model (`keras.models.Model`): The model in which to load the checkpoint.
        shard_files (`str` or `os.PathLike`): A list containing the sharded checkpoint names.
        ignore_mismatched_sizes`bool`, *optional`, defaults to `True`):
            Whether or not to ignore the mismatch between the sizes
        strict (`bool`, *optional*, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.

    Returns:
        Three lists, one for the missing layers, another one for the unexpected layers, and a last one for the
        mismatched layers.
    """
    unexpected_keys = set()
    all_missing_keys = []
    mismatched_keys = set()
    for shard_file in shard_files:
        missing_layers, unexpected_layers, mismatched_layers = load_tf_weights_from_safetensors(model, shard_file, ignore_mismatched_sizes=ignore_mismatched_sizes, _prefix=_prefix)
        all_missing_keys.append(set(missing_layers))
        unexpected_keys.update(unexpected_layers)
        mismatched_keys.update(mismatched_layers)
        gc.collect()
    missing_keys = set.intersection(*all_missing_keys)
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f'Error(s) in loading state_dict for {model.__class__.__name__}'
        if len(missing_keys) > 0:
            str_missing_keys = ','.join([f'"{k}"' for k in missing_keys])
            error_message += f'\nMissing key(s): {str_missing_keys}.'
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ','.join([f'"{k}"' for k in unexpected_keys])
            error_message += f'\nMissing key(s): {str_unexpected_keys}.'
        raise RuntimeError(error_message)
    return (missing_keys, unexpected_keys, mismatched_keys)