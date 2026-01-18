from __future__ import annotations
import abc
import contextlib, os.path, re
import enum
import itertools
import typing as T
from functools import lru_cache
from .. import coredata
from .. import mlog
from .. import mesonlib
from ..mesonlib import (
from ..arglist import CompilerArgs
def get_compiler_args_for_mode(self, mode: CompileCheckMode) -> T.List[str]:
    args: T.List[str] = []
    args += self.get_always_args()
    if mode is CompileCheckMode.COMPILE:
        args += self.get_compile_only_args()
    elif mode is CompileCheckMode.PREPROCESS:
        args += self.get_preprocess_only_args()
    else:
        assert mode is CompileCheckMode.LINK
    return args