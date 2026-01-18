from __future__ import annotations
import typing as T
import os
from pathlib import Path
from ..compilers import clike_debug_args, clike_optimization_args
from ...mesonlib import OptionKey
def get_module_incdir_args(self) -> T.Tuple[str]:
    return ('-module',)