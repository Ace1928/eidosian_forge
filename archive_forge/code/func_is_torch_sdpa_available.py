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
def is_torch_sdpa_available():
    if not is_torch_available():
        return False
    elif _torch_version == 'N/A':
        return False
    return version.parse(_torch_version) >= version.parse('2.1.1')