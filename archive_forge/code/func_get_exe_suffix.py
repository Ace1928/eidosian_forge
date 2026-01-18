from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
def get_exe_suffix(self) -> str:
    if self.is_windows() or self.is_cygwin():
        return 'exe'
    else:
        return ''