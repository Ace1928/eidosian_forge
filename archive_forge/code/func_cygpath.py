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
def cygpath(path: str) -> str:
    """Use :meth:`git.cmd.Git.polish_url` instead, that works on any environment."""
    path = str(path)
    if not path.startswith(('/cygdrive', '//', '/proc/cygdrive')):
        for regex, parser, recurse in _cygpath_parsers:
            match = regex.match(path)
            if match:
                path = parser(*match.groups())
                if recurse:
                    path = cygpath(path)
                break
        else:
            path = _cygexpath(None, path)
    return path