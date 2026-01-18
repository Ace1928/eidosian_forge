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
def remove_linkerlike_args(self, args: T.List[str]) -> T.List[str]:
    rm_exact = ('-headerpad_max_install_names',)
    rm_prefixes = ('-Wl,', '-L')
    rm_next = ('-L', '-framework')
    ret: T.List[str] = []
    iargs = iter(args)
    for arg in iargs:
        if arg in rm_exact:
            continue
        if arg.startswith(rm_prefixes) and arg not in rm_prefixes:
            continue
        if arg in rm_next:
            next(iargs)
            continue
        ret.append(arg)
    return ret