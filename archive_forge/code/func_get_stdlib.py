from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
def get_stdlib(self, language: str) -> T.Union[str, T.List[str]]:
    stdlib = self.properties[language + '_stdlib']
    if isinstance(stdlib, str):
        return stdlib
    assert isinstance(stdlib, list)
    for i in stdlib:
        assert isinstance(i, str)
    return stdlib