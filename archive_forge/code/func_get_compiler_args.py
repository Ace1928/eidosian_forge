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
def get_compiler_args(self) -> T.List[str]:
    args: T.List[str] = []
    if self.mod_name in boost_libraries:
        libdef = boost_libraries[self.mod_name]
        if self.static:
            args += libdef.static
        else:
            args += libdef.shared
        if self.mt:
            args += libdef.multi
        else:
            args += libdef.single
    return args