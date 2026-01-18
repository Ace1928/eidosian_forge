import contextlib
import errno
import getpass
import hashlib
import io
import logging
import os
import posixpath
import shutil
import stat
import sys
import sysconfig
import urllib.parse
from functools import partial
from io import StringIO
from itertools import filterfalse, tee, zip_longest
from pathlib import Path
from types import FunctionType, TracebackType
from typing import (
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.pyproject_hooks import BuildBackendHookCaller
from pip._vendor.tenacity import retry, stop_after_delay, wait_fixed
from pip import __version__
from pip._internal.exceptions import CommandError, ExternallyManagedEnvironment
from pip._internal.locations import get_major_minor_version
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.virtualenv import running_under_virtualenv
def normalize_version_info(py_version_info: Tuple[int, ...]) -> Tuple[int, int, int]:
    """
    Convert a tuple of ints representing a Python version to one of length
    three.

    :param py_version_info: a tuple of ints representing a Python version,
        or None to specify no version. The tuple can have any length.

    :return: a tuple of length three if `py_version_info` is non-None.
        Otherwise, return `py_version_info` unchanged (i.e. None).
    """
    if len(py_version_info) < 3:
        py_version_info += (3 - len(py_version_info)) * (0,)
    elif len(py_version_info) > 3:
        py_version_info = py_version_info[:3]
    return cast('VersionInfo', py_version_info)