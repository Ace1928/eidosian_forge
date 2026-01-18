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
def remove_color(cls, segments: Iterable['Segment']) -> Iterable['Segment']:
    """Remove all color from an iterable of segments.

        Args:
            segments (Iterable[Segment]): An iterable segments.

        Yields:
            Segment: Segments with colorless style.
        """
    cache: Dict[Style, Style] = {}
    for text, style, control in segments:
        if style:
            colorless_style = cache.get(style)
            if colorless_style is None:
                colorless_style = style.without_color
                cache[style] = colorless_style
            yield cls(text, colorless_style, control)
        else:
            yield cls(text, None, control)