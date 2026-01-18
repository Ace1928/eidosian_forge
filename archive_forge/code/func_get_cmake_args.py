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
def get_cmake_args(self) -> T.List[str]:
    args = ['-DCMAKE_TOOLCHAIN_FILE=' + self.toolchain_file.as_posix()]
    if self.preload_file is not None:
        args += ['-DMESON_PRELOAD_FILE=' + self.preload_file.as_posix()]
    return args