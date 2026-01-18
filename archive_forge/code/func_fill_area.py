from __future__ import annotations
from collections import defaultdict
from typing import TYPE_CHECKING, Callable
from prompt_toolkit.cache import FastDictCache
from prompt_toolkit.data_structures import Point
from prompt_toolkit.utils import get_cwidth
def fill_area(self, write_position: WritePosition, style: str='', after: bool=False) -> None:
    """
        Fill the content of this area, using the given `style`.
        The style is prepended before whatever was here before.
        """
    if not style.strip():
        return
    xmin = write_position.xpos
    xmax = write_position.xpos + write_position.width
    char_cache = _CHAR_CACHE
    data_buffer = self.data_buffer
    if after:
        append_style = ' ' + style
        prepend_style = ''
    else:
        append_style = ''
        prepend_style = style + ' '
    for y in range(write_position.ypos, write_position.ypos + write_position.height):
        row = data_buffer[y]
        for x in range(xmin, xmax):
            cell = row[x]
            row[x] = char_cache[cell.char, prepend_style + cell.style + append_style]