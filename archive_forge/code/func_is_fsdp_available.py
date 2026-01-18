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
def is_fsdp_available(min_version: str=FSDP_MIN_VERSION):
    return is_torch_available() and version.parse(_torch_version) >= version.parse(min_version)