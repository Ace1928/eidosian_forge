import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union
from packaging import version
from . import logging
@lru_cache()
def is_torch_tpu_available(check_device=True):
    """Checks if `torch_xla` is installed and potentially if a TPU is in the environment"""
    if not _torch_available:
        return False
    if importlib.util.find_spec('torch_xla') is not None:
        if check_device:
            try:
                import torch_xla.core.xla_model as xm
                _ = xm.xla_device()
                return True
            except RuntimeError:
                return False
        return True
    return False