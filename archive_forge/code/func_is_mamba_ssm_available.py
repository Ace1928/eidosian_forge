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
def is_mamba_ssm_available():
    if is_torch_available():
        import torch
        if not torch.cuda.is_available():
            return False
        else:
            return _is_package_available('mamba_ssm')
    return False