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
@staticmethod
def is_cmdline_option(compiler: 'Compiler', arg: str) -> bool:
    if compiler.get_argument_syntax() == 'msvc':
        return arg.startswith('/')
    else:
        return arg.startswith('-')