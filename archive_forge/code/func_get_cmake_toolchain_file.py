from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
def get_cmake_toolchain_file(self) -> T.Optional[Path]:
    if 'cmake_toolchain_file' not in self.properties:
        return None
    raw = self.properties['cmake_toolchain_file']
    assert isinstance(raw, str)
    cmake_toolchain_file = Path(raw)
    if not cmake_toolchain_file.is_absolute():
        raise EnvironmentException(f'cmake_toolchain_file ({raw}) is not absolute')
    return cmake_toolchain_file