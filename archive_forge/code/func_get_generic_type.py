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
def get_generic_type(base_type: t.Type, generic_base_type: t.Type[TValue]) -> t.Optional[t.Type[TValue]]:
    """Return the generic type arg derived from the generic_base_type type that is associated with the base_type type, if any, otherwise return None."""
    type_arg = t.get_args(base_type.__orig_bases__[0])[0]
    return None if isinstance(type_arg, generic_base_type) else type_arg