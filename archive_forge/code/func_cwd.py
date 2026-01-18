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
@contextlib.contextmanager
def cwd(new_dir: PathLike) -> Generator[PathLike, None, None]:
    """Context manager to temporarily change directory.

    This is similar to :func:`contextlib.chdir` introduced in Python 3.11, but the
    context manager object returned by a single call to this function is not reentrant.
    """
    old_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield new_dir
    finally:
        os.chdir(old_dir)