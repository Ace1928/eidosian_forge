from __future__ import annotations
from pathlib import Path
from .traceparser import CMakeTraceParser
from ..envconfig import CMakeSkipCompilerTest
from .common import language_map, cmake_get_generator_args
from .. import mlog
import shutil
import typing as T
from enum import Enum
from textwrap import dedent
def make_abs(exe: str) -> str:
    if Path(exe).is_absolute():
        return exe
    p = shutil.which(exe)
    if p is None:
        return exe
    return p