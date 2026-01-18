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
def gen_vs_module_defs_args(self, defsfile: str) -> T.List[str]:
    if not isinstance(defsfile, str):
        raise RuntimeError('Module definitions file should be str')
    if self.info.is_windows() or self.info.is_cygwin():
        return [defsfile]
    return []