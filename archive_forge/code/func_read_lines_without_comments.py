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
def read_lines_without_comments(path: str, remove_blank_lines: bool=False, optional: bool=False) -> list[str]:
    """
    Returns lines from the specified text file with comments removed.
    Comments are any content from a hash symbol to the end of a line.
    Any spaces immediately before a comment are also removed.
    """
    if optional and (not os.path.exists(path)):
        return []
    lines = read_text_file(path).splitlines()
    lines = [re.sub(' *#.*$', '', line) for line in lines]
    if remove_blank_lines:
        lines = [line for line in lines if line]
    return lines