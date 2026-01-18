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
def supported_warn_args(self, warn_args_by_version: T.Dict[str, T.List[str]]) -> T.List[str]:
    result: T.List[str] = []
    for version, warn_args in warn_args_by_version.items():
        if mesonlib.version_compare(self.version, '>=' + version):
            result += warn_args
    return result