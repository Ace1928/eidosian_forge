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
def join_path_native(a: PathLike, *p: PathLike) -> PathLike:
    """Like join_path, but makes sure an OS native path is returned.

    This is only needed to play it safe on Windows and to ensure nice paths that only
    use '\\'.
    """
    return to_native_path(join_path(a, *p))