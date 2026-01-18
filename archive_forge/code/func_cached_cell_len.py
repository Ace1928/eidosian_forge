import re
from functools import lru_cache
from typing import Callable, List
from ._cell_widths import CELL_WIDTHS
@lru_cache(4096)
def cached_cell_len(text: str) -> int:
    """Get the number of cells required to display text.

    This method always caches, which may use up a lot of memory. It is recommended to use
    `cell_len` over this method.

    Args:
        text (str): Text to display.

    Returns:
        int: Get the number of cells required to display text.
    """
    _get_size = get_character_cell_size
    total_size = sum((_get_size(character) for character in text))
    return total_size