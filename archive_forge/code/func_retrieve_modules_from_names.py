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
def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
    module_keys = {'.'.join(key.split('.')[:-1]) for key in names}
    module_keys = module_keys.union({'.'.join(key.split('.')[:-2]) for key in names if len(key) > 0 and key[-1].isdigit()})
    retrieved_modules = []
    for name, module in self.named_modules():
        if remove_prefix:
            _prefix = f'{self.base_model_prefix}.'
            name = name[len(_prefix):] if name.startswith(_prefix) else name
        elif add_prefix:
            name = '.'.join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix
        if name in module_keys:
            retrieved_modules.append(module)
    return retrieved_modules