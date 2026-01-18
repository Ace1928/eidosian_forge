from typing import Any, Iterator, Optional, Sequence
from ..utils.base64 import base64, unbase64
from .connection import (
def offset_to_cursor(offset: int) -> ConnectionCursor:
    """Create the cursor string from an offset."""
    return base64(f'{PREFIX}{offset}')