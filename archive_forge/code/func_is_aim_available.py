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
def is_aim_available():
    package_exists = _is_package_available('aim')
    if package_exists:
        aim_version = version.parse(importlib.metadata.version('aim'))
        return compare_versions(aim_version, '<', '4.0.0')
    return False