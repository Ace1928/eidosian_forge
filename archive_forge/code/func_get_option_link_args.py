from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def get_option_link_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
    return []