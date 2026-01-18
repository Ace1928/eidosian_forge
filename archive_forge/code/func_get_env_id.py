import contextlib
import copy
import difflib
import importlib
import importlib.util
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import (
import numpy as np
from gym.wrappers import (
from gym.wrappers.compatibility import EnvCompatibility
from gym.wrappers.env_checker import PassiveEnvChecker
from gym import Env, error, logger
def get_env_id(ns: Optional[str], name: str, version: Optional[int]) -> str:
    """Get the full env ID given a name and (optional) version and namespace. Inverse of :meth:`parse_env_id`.

    Args:
        ns: The environment namespace
        name: The environment name
        version: The environment version

    Returns:
        The environment id
    """
    full_name = name
    if version is not None:
        full_name += f'-v{version}'
    if ns is not None:
        full_name = ns + '/' + full_name
    return full_name