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
def fix_python_name(self, tags: T.List[str]) -> T.List[str]:
    other_tags: T.List[str] = []
    m_cur = BoostLibraryFile.reg_python_mod_split.match(self.mod_name)
    cur_name = m_cur.group(1)
    cur_vers = m_cur.group(2)

    def update_vers(new_vers: str) -> None:
        nonlocal cur_vers
        new_vers = new_vers.replace('_', '')
        new_vers = new_vers.replace('.', '')
        if not new_vers.isdigit():
            return
        if len(new_vers) > len(cur_vers):
            cur_vers = new_vers
    for i in tags:
        if i.startswith('py'):
            update_vers(i[2:])
        elif i.isdigit():
            update_vers(i)
        elif len(i) >= 3 and i[0].isdigit and i[2].isdigit() and (i[1] == '.'):
            update_vers(i)
        else:
            other_tags += [i]
    self.mod_name = cur_name + cur_vers
    return other_tags