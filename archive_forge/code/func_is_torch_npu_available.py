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
def is_torch_npu_available(check_device=False):
    """Checks if `torch_npu` is installed and potentially if a NPU is in the environment"""
    if not _torch_available or importlib.util.find_spec('torch_npu') is None:
        return False
    import torch
    import torch_npu
    if check_device:
        try:
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, 'npu') and torch.npu.is_available()