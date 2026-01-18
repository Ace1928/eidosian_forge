from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def sanitizer_link_args(self, value: str) -> T.List[str]:
    return []