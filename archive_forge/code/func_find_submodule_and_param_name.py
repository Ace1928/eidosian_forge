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
def find_submodule_and_param_name(model, long_key, start_prefix):
    """
    A helper util to find the last sub-module and the param/buffer name. If `start_prefix` is supplied it'll be removed
    from the start of the key
    """
    if len(start_prefix) > 0 and long_key.startswith(start_prefix):
        long_key = '.'.join(long_key.split('.')[1:])
    split_key = long_key.split('.')
    submodule = model
    while len(split_key) > 1:
        if hasattr(submodule, split_key[0]):
            submodule = getattr(submodule, split_key[0])
            del split_key[0]
        else:
            submodule = None
            break
    if submodule == model:
        submodule = None
    return (submodule, split_key[0])