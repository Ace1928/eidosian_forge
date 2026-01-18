from __future__ import annotations
from . import (
@staticmethod
def is_content_root(path: str) -> bool:
    """Return True if the given path is a content root for this provider."""
    return False