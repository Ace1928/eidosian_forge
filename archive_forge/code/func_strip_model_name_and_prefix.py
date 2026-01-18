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
def strip_model_name_and_prefix(name, _prefix=None):
    if _prefix is not None and name.startswith(_prefix):
        name = name[len(_prefix):]
        if name.startswith('/'):
            name = name[1:]
    if 'model.' not in name and len(name.split('/')) > 1:
        name = '/'.join(name.split('/')[1:])
    return name