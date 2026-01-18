from __future__ import annotations
import subprocess, os.path
import typing as T
from ..mesonlib import EnvironmentException
from .compilers import Compiler, clike_debug_args
def get_header_import_args(self, headername: str) -> T.List[str]:
    return ['-import-objc-header', headername]