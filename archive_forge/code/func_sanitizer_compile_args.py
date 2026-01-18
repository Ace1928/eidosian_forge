from __future__ import annotations
import abc
import functools
import os
import multiprocessing
import pathlib
import re
import subprocess
import typing as T
from ... import mesonlib
from ... import mlog
from ...mesonlib import OptionKey
from mesonbuild.compilers.compilers import CompileCheckMode
def sanitizer_compile_args(self, value: str) -> T.List[str]:
    if value == 'none':
        return []
    args = ['-fsanitize=' + value]
    if 'address' in value:
        args.append('-fno-omit-frame-pointer')
    return args