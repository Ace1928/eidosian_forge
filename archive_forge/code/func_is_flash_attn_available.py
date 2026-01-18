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
def is_flash_attn_available():
    logger.warning('Using `is_flash_attn_available` is deprecated and will be removed in v4.38. Please use `is_flash_attn_2_available` instead.')
    return is_flash_attn_2_available()