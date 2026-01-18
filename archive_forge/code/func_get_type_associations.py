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
def get_type_associations(base_type: t.Type[TBase], generic_base_type: t.Type[TValue]) -> list[tuple[t.Type[TValue], t.Type[TBase]]]:
    """Create and return a list of tuples associating generic_base_type derived types with a corresponding base_type derived type."""
    return [item for item in [(get_generic_type(sc_type, generic_base_type), sc_type) for sc_type in get_subclasses(base_type)] if item[1]]