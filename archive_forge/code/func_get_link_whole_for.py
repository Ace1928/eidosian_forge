from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def get_link_whole_for(self, args: T.List[str]) -> T.List[str]:
    raise EnvironmentException(f'Linker {self.id} does not support link_whole')