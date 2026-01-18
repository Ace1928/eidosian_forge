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
def is_torch_bf16_available():
    warnings.warn("The util is_torch_bf16_available is deprecated, please use is_torch_bf16_gpu_available or is_torch_bf16_cpu_available instead according to whether it's used with cpu or gpu", FutureWarning)
    return is_torch_bf16_gpu_available()