from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
def get_cmake_skip_compiler_test(self) -> CMakeSkipCompilerTest:
    if 'cmake_skip_compiler_test' not in self.properties:
        return CMakeSkipCompilerTest.DEP_ONLY
    raw = self.properties['cmake_skip_compiler_test']
    assert isinstance(raw, str)
    try:
        return CMakeSkipCompilerTest(raw)
    except ValueError:
        raise EnvironmentException('"{}" is not a valid value for cmake_skip_compiler_test. Supported values are {}'.format(raw, [e.value for e in CMakeSkipCompilerTest]))