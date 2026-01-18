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
def flax_shard_checkpoint(params, max_shard_size='10GB'):
    """
    Splits a model state dictionary in sub-checkpoints so that the final size of each sub-checkpoint does not exceed a
    given size. The sub-checkpoints are determined by iterating through the `state_dict` in the order of its keys, so
    there is no optimization made to make each sub-checkpoint as close as possible to the maximum size passed. For
    example, if the limit is 10GB and we have weights of sizes [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] they will get sharded as
    [6GB], [6+2GB], [6+2+2GB] and not [6+2+2GB], [6+2GB], [6GB].

    <Tip warning={true}>

    If one of the model's weight is bigger that `max_shard_size`, it will end up in its own sub-checkpoint which will
    have a size greater than `max_shard_size`.

    </Tip>

    Args:
        params (`Union[Dict, FrozenDict]`): A `PyTree` of model parameters.
        max_shard_size (`int` or `str`, *optional*, defaults to `"10GB"`):
            The maximum size of each sub-checkpoint. If expressed as a string, needs to be digits followed by a unit
            (like `"5MB"`).
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)
    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0
    weights = flatten_dict(params, sep='/')
    for item in weights:
        weight_size = weights[item].size * dtype_byte_size(weights[item].dtype)
        if current_block_size + weight_size > max_shard_size:
            sharded_state_dicts.append(current_block)
            current_block = {}
            current_block_size = 0
        current_block[item] = weights[item]
        current_block_size += weight_size
        total_size += weight_size
    sharded_state_dicts.append(current_block)
    if len(sharded_state_dicts) == 1:
        return ({FLAX_WEIGHTS_NAME: sharded_state_dicts[0]}, None)
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        shard_file = FLAX_WEIGHTS_NAME.replace('.msgpack', f'-{idx + 1:05d}-of-{len(sharded_state_dicts):05d}.msgpack')
        shards[shard_file] = shard
        for weight_name in shard.keys():
            weight_map[weight_name] = shard_file
    metadata = {'total_size': total_size}
    index = {'metadata': metadata, 'weight_map': weight_map}
    return (shards, index)