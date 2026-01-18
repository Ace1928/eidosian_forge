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
def get_largefile_args(self) -> T.List[str]:
    """Enable transparent large-file-support for 32-bit UNIX systems"""
    if not (self.get_argument_syntax() == 'msvc' or self.info.is_darwin()):
        return ['-D_FILE_OFFSET_BITS=64']
    return []