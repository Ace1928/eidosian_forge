from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def get_link_debugfile_args(self, targetfile: str) -> T.List[str]:
    return []