import importlib.metadata
import platform
import sys
import warnings
from typing import Any, Dict
from .. import __version__, constants
def get_fastcore_version() -> str:
    return _get_version('fastcore')