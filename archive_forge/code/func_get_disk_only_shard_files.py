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
def get_disk_only_shard_files(device_map, sharded_metadata, start_prefix):
    """
    Returns the list of shard files containing only weights offloaded to disk.
    """
    weight_map = {p[len(start_prefix):]: v for p, v in sharded_metadata['weight_map'].items() if p.startswith(start_prefix)}
    files_content = collections.defaultdict(list)
    for weight_name, filename in weight_map.items():
        while len(weight_name) > 0 and weight_name not in device_map:
            weight_name = '.'.join(weight_name.split('.')[:-1])
        files_content[filename].append(device_map[weight_name])
    return [fname for fname, devices in files_content.items() if set(devices) == {'disk'}]