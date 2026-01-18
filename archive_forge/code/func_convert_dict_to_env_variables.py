import logging
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from shutil import which
from typing import List, Optional
import torch
from packaging.version import parse
def convert_dict_to_env_variables(current_env: dict):
    """
    Verifies that all keys and values in `current_env` do not contain illegal keys or values, and returns a list of
    strings as the result.

    Example:
    ```python
    >>> from accelerate.utils.environment import verify_env

    >>> env = {"ACCELERATE_DEBUG_MODE": "1", "BAD_ENV_NAME": "<mything", "OTHER_ENV": "2"}
    >>> valid_env_items = verify_env(env)
    >>> print(valid_env_items)
    ["ACCELERATE_DEBUG_MODE=1
", "OTHER_ENV=2
"]
    ```
    """
    forbidden_chars = [';', '\n', '<', '>', ' ']
    valid_env_items = []
    for key, value in current_env.items():
        if all((char not in key + value for char in forbidden_chars)) and len(key) >= 1 and (len(value) >= 1):
            valid_env_items.append(f'{key}={value}\n')
        else:
            logger.warning(f'WARNING: Skipping {key}={value} as it contains forbidden characters or missing values.')
    return valid_env_items