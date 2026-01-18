from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
def get_pkg_config_libdir(self) -> T.Optional[T.List[str]]:
    p = self.properties.get('pkg_config_libdir', None)
    if p is None:
        return p
    res = mesonlib.listify(p)
    for i in res:
        assert isinstance(i, str)
    return res