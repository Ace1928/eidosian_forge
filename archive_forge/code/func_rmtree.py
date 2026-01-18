from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
def rmtree(path: PathLike) -> None:
    """Remove the given directory tree recursively.

    :note: We use :func:`shutil.rmtree` but adjust its behaviour to see whether files
        that couldn't be deleted are read-only. Windows will not remove them in that
        case.
    """

    def handler(function: Callable, path: PathLike, _excinfo: Any) -> None:
        """Callback for :func:`shutil.rmtree`. Works either as ``onexc`` or ``onerror``."""
        os.chmod(path, stat.S_IWUSR)
        try:
            function(path)
        except PermissionError as ex:
            if HIDE_WINDOWS_KNOWN_ERRORS:
                from unittest import SkipTest
                raise SkipTest(f'FIXME: fails with: PermissionError\n  {ex}') from ex
            raise
    if os.name != 'nt':
        shutil.rmtree(path)
    elif sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=handler)
    else:
        shutil.rmtree(path, onerror=handler)