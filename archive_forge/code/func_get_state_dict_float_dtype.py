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
def get_state_dict_float_dtype(state_dict):
    """
    Returns the first found floating dtype in `state_dict` or asserts if none were found.
    """
    for t in state_dict.values():
        if t.is_floating_point():
            return t.dtype
    raise ValueError("couldn't find any floating point dtypes in state_dict")