from __future__ import annotations
from ..mesonlib import MesonException, OptionKey
from .. import mlog
from pathlib import Path
import typing as T
def check_cmake_args(args: T.List[str]) -> T.List[str]:
    res: T.List[str] = []
    dis = ['-D' + x for x in blacklist_cmake_defs]
    assert dis
    for i in args:
        if any((i.startswith(x) for x in dis)):
            mlog.warning('Setting', mlog.bold(i), 'is not supported. See the meson docs for cross compilation support:')
            mlog.warning('  - URL: https://mesonbuild.com/CMake-module.html#cross-compilation')
            mlog.warning('  --> Ignoring this option')
            continue
        res += [i]
    return res