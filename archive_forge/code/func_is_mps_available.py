import importlib
import importlib.metadata
import os
import warnings
from functools import lru_cache
import torch
from packaging import version
from packaging.version import parse
from .environment import parse_flag_from_env, str_to_bool
from .versions import compare_versions, is_torch_version
def is_mps_available():
    return is_torch_version('>=', '1.12') and torch.backends.mps.is_available() and torch.backends.mps.is_built()