from __future__ import annotations
from typing import Tuple, Union
def get_version_string() -> str:
    if isinstance(version_tuple[-1], str):
        return '.'.join(map(str, version_tuple[:-1])) + version_tuple[-1]
    return '.'.join(map(str, version_tuple))