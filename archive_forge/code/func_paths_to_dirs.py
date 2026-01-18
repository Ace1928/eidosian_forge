from __future__ import annotations
import abc
import collections.abc as c
import enum
import fcntl
import importlib.util
import inspect
import json
import keyword
import os
import platform
import pkgutil
import random
import re
import shutil
import stat
import string
import subprocess
import sys
import time
import functools
import shlex
import typing as t
import warnings
from struct import unpack, pack
from termios import TIOCGWINSZ
from .locale_util import (
from .encoding import (
from .io import (
from .thread import (
from .constants import (
def paths_to_dirs(paths: list[str]) -> list[str]:
    """Returns a list of directories extracted from the given list of paths."""
    dir_names = set()
    for path in paths:
        while True:
            path = os.path.dirname(path)
            if not path or path == os.path.sep:
                break
            dir_names.add(path + os.path.sep)
    return sorted(dir_names)