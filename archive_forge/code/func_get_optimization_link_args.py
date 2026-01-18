from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def get_optimization_link_args(self, optimization_level: str) -> T.List[str]:
    return []