from __future__ import annotations
from vine import transform
from .message import AsyncMessage
def list_first(rs):
    """Get the first item in a list, or None if list empty."""
    return rs[0] if len(rs) == 1 else None