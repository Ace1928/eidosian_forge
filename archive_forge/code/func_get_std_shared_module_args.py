from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def get_std_shared_module_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
    return self.get_std_shared_lib_link_args()