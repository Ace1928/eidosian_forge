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
def detect_libraries(self, libdir: Path) -> T.List[BoostLibraryFile]:
    libs: T.Set[BoostLibraryFile] = set()
    for i in libdir.iterdir():
        if not i.is_file():
            continue
        if not any((i.name.startswith(x) for x in ['libboost_', 'boost_'])):
            continue
        if i.name.endswith('.pdb'):
            continue
        try:
            libs.add(BoostLibraryFile(i.resolve()))
        except UnknownFileException as e:
            mlog.warning('Boost: ignoring unknown file {} under lib directory'.format(e.path.name))
    return [x for x in libs if x.is_boost()]