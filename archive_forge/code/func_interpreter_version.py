import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def interpreter_version(*, warn: bool=False) -> str:
    """
    Returns the version of the running interpreter.
    """
    version = _get_config_var('py_version_nodot', warn=warn)
    if version:
        version = str(version)
    else:
        version = _version_nodot(sys.version_info[:2])
    return version