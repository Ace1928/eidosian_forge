from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def bitcode_args(self) -> T.List[str]:
    raise MesonException("This linker doesn't support bitcode bundles")