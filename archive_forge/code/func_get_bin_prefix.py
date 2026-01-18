import functools
import logging
import os
import pathlib
import sys
import sysconfig
from typing import Any, Dict, Generator, Optional, Tuple
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.virtualenv import running_under_virtualenv
from . import _sysconfig
from .base import (
def get_bin_prefix() -> str:
    new = _sysconfig.get_bin_prefix()
    if _USE_SYSCONFIG:
        return new
    old = _distutils.get_bin_prefix()
    if _warn_if_mismatch(pathlib.Path(old), pathlib.Path(new), key='bin_prefix'):
        _log_context()
    return old