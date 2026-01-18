import re
from functools import lru_cache
from typing import Callable, List
from ._cell_widths import CELL_WIDTHS
def chop_cells(text: str, max_size: int, position: int=0) -> List[str]:
    """Break text in to equal (cell) length strings, returning the characters in reverse
    order"""
    _get_character_cell_size = get_character_cell_size
    characters = [(character, _get_character_cell_size(character)) for character in text]
    total_size = position
    lines: List[List[str]] = [[]]
    append = lines[-1].append
    for character, size in reversed(characters):
        if total_size + size > max_size:
            lines.append([character])
            append = lines[-1].append
            total_size = size
        else:
            total_size += size
            append(character)
    return [''.join(line) for line in lines]