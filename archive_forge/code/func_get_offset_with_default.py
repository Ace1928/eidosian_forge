from typing import Any, Iterator, Optional, Sequence
from ..utils.base64 import base64, unbase64
from .connection import (
def get_offset_with_default(cursor: Optional[ConnectionCursor]=None, default_offset: int=0) -> int:
    """Get offset from a given cursor and a default.

    Given an optional cursor and a default offset, return the offset to use;
    if the cursor contains a valid offset, that will be used,
    otherwise it will be the default.
    """
    if not isinstance(cursor, str):
        return default_offset
    offset = cursor_to_offset(cursor)
    return default_offset if offset is None else offset