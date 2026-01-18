from enum import IntEnum
from functools import lru_cache
from itertools import filterfalse
from logging import getLogger
from operator import attrgetter
from typing import (
from .cells import (
from .repr import Result, rich_repr
from .style import Style
@classmethod
def align_bottom(cls: Type['Segment'], lines: List[List['Segment']], width: int, height: int, style: Style, new_lines: bool=False) -> List[List['Segment']]:
    """Aligns render to bottom (adds extra lines above as required).

        Args:
            lines (List[List[Segment]]): A list of lines.
            width (int): Desired width.
            height (int, optional): Desired height or None for no change.
            style (Style): Style of any padding added. Defaults to None.
            new_lines (bool, optional): Padded lines should include "
". Defaults to False.

        Returns:
            List[List[Segment]]: New list of lines.
        """
    extra_lines = height - len(lines)
    if not extra_lines:
        return lines[:]
    lines = lines[:height]
    blank = cls(' ' * width + '\n', style) if new_lines else cls(' ' * width, style)
    lines = [[blank]] * extra_lines + lines
    return lines