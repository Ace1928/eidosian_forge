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
def get_include_args(self, path: str, is_system: bool) -> T.List[str]:
    if not path:
        path = '.'
    if is_system:
        return ['-isystem' + path]
    return ['-I' + path]