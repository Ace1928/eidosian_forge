from __future__ import annotations
import os
import typing as T
from ... import mesonlib
from ..compilers import CompileCheckMode
from .gnu import GnuLikeCompiler
from .visualstudio import VisualStudioLikeCompiler
def get_pch_use_args(self, pch_dir: str, header: str) -> T.List[str]:
    return ['-pch', '-pch_dir', os.path.join(pch_dir), '-x', self.lang_header, '-include', header, '-x', 'none']