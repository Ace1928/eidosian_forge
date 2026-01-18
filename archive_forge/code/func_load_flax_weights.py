import gc
import json
import os
import re
import warnings
from functools import partial
from pickle import UnpicklingError
from typing import Any, Dict, Optional, Set, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import msgpack.exceptions
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.serialization import from_bytes, to_bytes
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import FlaxGenerationMixin, GenerationConfig
from .modeling_flax_pytorch_utils import load_pytorch_checkpoint_in_flax_state_dict
from .utils import (
from .utils.hub import convert_file_size_to_int, get_checkpoint_shard_files
from .utils.import_utils import is_safetensors_available
@classmethod
def load_flax_weights(cls, resolved_archive_file):
    try:
        if resolved_archive_file.endswith('.safetensors'):
            state = safe_load_file(resolved_archive_file)
            state = unflatten_dict(state, sep='.')
        else:
            with open(resolved_archive_file, 'rb') as state_f:
                state = from_bytes(cls, state_f.read())
    except (UnpicklingError, msgpack.exceptions.ExtraData) as e:
        try:
            with open(resolved_archive_file) as f:
                if f.read().startswith('version'):
                    raise OSError('You seem to have cloned a repository without having git-lfs installed. Please install git-lfs and run `git lfs install` followed by `git lfs pull` in the folder you cloned.')
                else:
                    raise ValueError from e
        except (UnicodeDecodeError, ValueError):
            raise EnvironmentError(f'Unable to convert {resolved_archive_file} to Flax deserializable object. ')
    return state