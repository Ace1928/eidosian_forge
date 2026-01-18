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
def is_flash_attn_greater_or_equal_2_10():
    if not _is_package_available('flash_attn'):
        return False
    return version.parse(importlib.metadata.version('flash_attn')) >= version.parse('2.1.0')