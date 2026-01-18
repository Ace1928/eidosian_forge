from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def get_pie_link_args(self) -> T.List[str]:
    raise EnvironmentException(f'Linker {self.id} does not support position-independent executable')