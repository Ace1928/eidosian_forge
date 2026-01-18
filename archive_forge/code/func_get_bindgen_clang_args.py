from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
def get_bindgen_clang_args(self) -> T.List[str]:
    value = mesonlib.listify(self.properties.get('bindgen_clang_arguments', []))
    if not all((isinstance(v, str) for v in value)):
        raise EnvironmentException('bindgen_clang_arguments must be a string or an array of strings')
    return T.cast('T.List[str]', value)