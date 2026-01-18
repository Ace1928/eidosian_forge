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
def display_path(path: str) -> str:
    """Gives the display value for a given path, making it relative to cwd
    if possible."""
    path = os.path.normcase(os.path.abspath(path))
    if path.startswith(os.getcwd() + os.path.sep):
        path = '.' + path[len(os.getcwd()):]
    return path