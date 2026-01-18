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
def rmtree_errorhandler(func: FunctionType, path: Path, exc_info: Union[ExcInfo, BaseException], *, onexc: OnExc=_onerror_reraise) -> None:
    """
    `rmtree` error handler to 'force' a file remove (i.e. like `rm -f`).

    * If a file is readonly then it's write flag is set and operation is
      retried.

    * `onerror` is the original callback from `rmtree(... onerror=onerror)`
      that is chained at the end if the "rm -f" still fails.
    """
    try:
        st_mode = os.stat(path).st_mode
    except OSError:
        return
    if not st_mode & stat.S_IWRITE:
        try:
            os.chmod(path, st_mode | stat.S_IWRITE)
        except OSError:
            pass
        else:
            try:
                func(path)
                return
            except OSError:
                pass
    if not isinstance(exc_info, BaseException):
        _, exc_info, _ = exc_info
    onexc(func, path, exc_info)