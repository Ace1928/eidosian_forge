from __future__ import annotations
import os
import typing as T
from ... import mesonlib
from ..compilers import CompileCheckMode
from .gnu import GnuLikeCompiler
from .visualstudio import VisualStudioLikeCompiler
def get_pch_name(self, name: str) -> str:
    return os.path.basename(name) + '.' + self.get_pch_suffix()