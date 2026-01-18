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
def import_plugins(directory: str, root: t.Optional[str]=None) -> None:
    """
    Import plugins from the given directory relative to the given root.
    If the root is not provided, the 'lib' directory for the test runner will be used.
    """
    if root is None:
        root = os.path.dirname(__file__)
    path = os.path.join(root, directory)
    package = __name__.rsplit('.', 1)[0]
    prefix = '%s.%s.' % (package, directory.replace(os.path.sep, '.'))
    for _module_loader, name, _ispkg in pkgutil.iter_modules([path], prefix=prefix):
        module_path = os.path.join(root, name[len(package) + 1:].replace('.', os.path.sep) + '.py')
        load_module(module_path, name)