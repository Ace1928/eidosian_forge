from __future__ import annotations
import re
import dataclasses
import functools
import typing as T
from pathlib import Path
from .. import mlog
from .. import mesonlib
from .base import DependencyException, SystemDependency
from .detect import packages
from .pkgconfig import PkgConfigDependency
from .misc import threads_factory
def mod_name_matches(self, mod_name: str) -> bool:
    if self.mod_name == mod_name:
        return True
    if not self.is_python_lib():
        return False
    m_cur = BoostLibraryFile.reg_python_mod_split.match(self.mod_name)
    m_arg = BoostLibraryFile.reg_python_mod_split.match(mod_name)
    if not m_cur or not m_arg:
        return False
    if m_cur.group(1) != m_arg.group(1):
        return False
    cur_vers = m_cur.group(2)
    arg_vers = m_arg.group(2)
    if not arg_vers:
        arg_vers = '2'
    return cur_vers.startswith(arg_vers)