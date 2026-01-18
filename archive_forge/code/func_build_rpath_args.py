from __future__ import annotations
import typing as T
from ...mesonlib import EnvironmentException, MesonException, is_windows
def build_rpath_args(self, env: 'Environment', build_dir: str, from_dir: str, rpath_paths: T.Tuple[str, ...], build_rpath: str, install_rpath: str) -> T.Tuple[T.List[str], T.Set[bytes]]:
    return ([], set())