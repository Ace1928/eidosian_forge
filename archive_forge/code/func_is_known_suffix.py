from __future__ import annotations
import abc
import contextlib, os.path, re
import enum
import itertools
import typing as T
from functools import lru_cache
from .. import coredata
from .. import mlog
from .. import mesonlib
from ..mesonlib import (
from ..arglist import CompilerArgs
def is_known_suffix(fname: 'mesonlib.FileOrString') -> bool:
    if isinstance(fname, mesonlib.File):
        fname = fname.fname
    suffix = fname.split('.')[-1]
    return suffix in all_suffixes