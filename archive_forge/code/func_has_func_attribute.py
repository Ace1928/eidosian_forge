from __future__ import annotations
import abc
import os
import typing as T
from ... import arglist
from ... import mesonlib
from ... import mlog
from mesonbuild.compilers.compilers import CompileCheckMode
def has_func_attribute(self, name: str, env: 'Environment') -> T.Tuple[bool, bool]:
    return (name in {'dllimport', 'dllexport'}, False)