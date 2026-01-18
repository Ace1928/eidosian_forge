import importlib.metadata
import platform
import sys
import warnings
from typing import Any, Dict
from .. import __version__, constants
def is_numpy_available() -> bool:
    return is_package_available('numpy')