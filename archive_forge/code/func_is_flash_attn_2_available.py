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
def is_flash_attn_2_available():
    if not is_torch_available():
        return False
    if not _is_package_available('flash_attn'):
        return False
    import torch
    if not torch.cuda.is_available():
        return False
    if torch.version.cuda:
        return version.parse(importlib.metadata.version('flash_attn')) >= version.parse('2.1.0')
    elif torch.version.hip:
        return version.parse(importlib.metadata.version('flash_attn')) >= version.parse('2.0.4')
    else:
        return False