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
def has_arguments(self, args: T.List[str], env: 'Environment', code: str, mode: CompileCheckMode) -> T.Tuple[bool, bool]:
    with self._build_wrapper(code, env, args, None, mode) as p:
        result = p.returncode == 0
        if self.language in {'cpp', 'objcpp'} and 'is valid for C/ObjC' in p.stderr:
            result = False
        if self.language in {'c', 'objc'} and 'is valid for C++/ObjC++' in p.stderr:
            result = False
    return (result, p.cached)