from __future__ import annotations
import logging # isort:skip
from .. import __version__
def is_full_release(version: str | None=None) -> bool:
    import re
    version = version or __version__
    VERSION_PAT = re.compile('^(\\d+\\.\\d+\\.\\d+)$')
    return bool(VERSION_PAT.match(version))