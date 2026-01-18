import importlib.metadata
import platform
import sys
import warnings
from typing import Any, Dict
from .. import __version__, constants
def is_package_available(package_name: str) -> bool:
    return _get_version(package_name) != 'N/A'