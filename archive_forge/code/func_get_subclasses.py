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
def get_subclasses(class_type: t.Type[C]) -> list[t.Type[C]]:
    """Returns a list of types that are concrete subclasses of the given type."""
    subclasses: set[t.Type[C]] = set()
    queue: list[t.Type[C]] = [class_type]
    while queue:
        parent = queue.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                if not inspect.isabstract(child):
                    subclasses.add(child)
                queue.append(child)
    return sorted(subclasses, key=lambda sc: sc.__name__)