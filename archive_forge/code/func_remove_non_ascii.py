from __future__ import annotations
from typing import TYPE_CHECKING, cast
def remove_non_ascii(s: str) -> str:
    """
    Remove non-ascii characters in a file. Needed when support for non-ASCII
    is not available.

    Args:
        s (str): Input string

    Returns:
        String with all non-ascii characters removed.
    """
    return ''.join((i for i in s if ord(i) < 128))