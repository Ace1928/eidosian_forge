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
def get_builtin_define(self, define: str) -> T.Optional[str]:
    if define in self.defines:
        return self.defines[define]
    return None