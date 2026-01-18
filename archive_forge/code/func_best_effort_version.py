import re
from pathlib import Path
from typing import Union
from .extern import packaging
def best_effort_version(version: str) -> str:
    """Convert an arbitrary string into a version-like string.
    Fallback when ``safe_version`` is not safe enough.
    >>> best_effort_version("v0.2 beta")
    '0.2b0'
    >>> best_effort_version("ubuntu lts")
    '0.dev0+sanitized.ubuntu.lts'
    >>> best_effort_version("0.23ubuntu1")
    '0.23.dev0+sanitized.ubuntu1'
    >>> best_effort_version("0.23-")
    '0.23.dev0+sanitized'
    >>> best_effort_version("0.-_")
    '0.dev0+sanitized'
    >>> best_effort_version("42.+?1")
    '42.dev0+sanitized.1'
    """
    try:
        return safe_version(version)
    except packaging.version.InvalidVersion:
        v = version.replace(' ', '.')
        match = _PEP440_FALLBACK.search(v)
        if match:
            safe = match['safe']
            rest = v[len(safe):]
        else:
            safe = '0'
            rest = version
        safe_rest = _NON_ALPHANUMERIC.sub('.', rest).strip('.')
        local = f'sanitized.{safe_rest}'.strip('.')
        return safe_version(f'{safe}.dev0+{local}')