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
def append_replace_return_docstrings(model_class, output_type, config_class):
    model_class.__call__ = copy_func(model_class.__call__)
    model_class.__call__ = replace_return_docstrings(output_type=output_type, config_class=config_class)(model_class.__call__)