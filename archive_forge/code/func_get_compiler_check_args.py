from __future__ import annotations
import os
import typing as T
from ... import mesonlib
from ..compilers import CompileCheckMode
from .gnu import GnuLikeCompiler
from .visualstudio import VisualStudioLikeCompiler
def get_compiler_check_args(self, mode: CompileCheckMode) -> T.List[str]:
    args = super().get_compiler_check_args(mode)
    if mode is not CompileCheckMode.LINK:
        args.extend(['/Qdiag-error:10006', '/Qdiag-error:10148', '/Qdiag-error:10155', '/Qdiag-error:10156', '/Qdiag-error:10157', '/Qdiag-error:10158'])
    return args