from __future__ import annotations
import enum
import os.path
import string
import typing as T
from .. import coredata
from .. import mlog
from ..mesonlib import (
from .compilers import Compiler
def is_xcompiler_flag_glued(flag: str) -> bool:
    return flag.startswith('-Xcompiler=')