from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def get_std_shared_lib_link_args(self) -> T.List[str]:
    return []