from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def get_linker_output_args(self, outputname: str) -> T.List[str]:
    return []