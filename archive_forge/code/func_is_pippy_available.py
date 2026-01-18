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
def is_pippy_available():
    package_exists = _is_package_available('pippy', 'torchpippy')
    if package_exists:
        pippy_version = version.parse(importlib.metadata.version('torchpippy'))
        return compare_versions(pippy_version, '>', '0.1.1')
    return False