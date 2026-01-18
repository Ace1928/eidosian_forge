import importlib.metadata
import platform
import sys
import warnings
from typing import Any, Dict
from .. import __version__, constants
def get_hf_hub_version() -> str:
    return __version__