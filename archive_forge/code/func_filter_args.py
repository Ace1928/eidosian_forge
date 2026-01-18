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
def filter_args(args: list[str], filters: dict[str, int]) -> list[str]:
    """Return a filtered version of the given command line arguments."""
    remaining = 0
    result = []
    for arg in args:
        if not arg.startswith('-') and remaining:
            remaining -= 1
            continue
        remaining = 0
        parts = arg.split('=', 1)
        key = parts[0]
        if key in filters:
            remaining = filters[key] - len(parts) + 1
            continue
        result.append(arg)
    return result