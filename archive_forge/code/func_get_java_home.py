from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
def get_java_home(self) -> T.Optional[Path]:
    value = T.cast('T.Optional[str]', self.properties.get('java_home'))
    return Path(value) if value else None