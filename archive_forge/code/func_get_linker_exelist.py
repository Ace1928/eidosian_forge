from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def get_linker_exelist(self) -> T.List[str]:
    return self.exelist.copy()