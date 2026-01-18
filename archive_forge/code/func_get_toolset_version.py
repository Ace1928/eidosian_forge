from __future__ import annotations
import os
import typing as T
from ... import mesonlib
from ..compilers import CompileCheckMode
from .gnu import GnuLikeCompiler
from .visualstudio import VisualStudioLikeCompiler
def get_toolset_version(self) -> T.Optional[str]:
    _, _, err = mesonlib.Popen_safe(['cl.exe'])
    v1, v2, *_ = mesonlib.search_version(err).split('.')
    version = int(v1 + v2)
    return self._calculate_toolset_version(version)