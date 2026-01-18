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
def detect_split_root(self, inc_dir: Path, lib_dir: Path) -> None:
    boost_inc_dir = None
    for j in [inc_dir / 'version.hpp', inc_dir / 'boost' / 'version.hpp']:
        if j.is_file():
            boost_inc_dir = self._include_dir_from_version_header(j)
            break
    if not boost_inc_dir:
        self.is_found = False
        return
    self.is_found = self.run_check([boost_inc_dir], [lib_dir])