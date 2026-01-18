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
def strip_links(cls, segments: Iterable['Segment']) -> Iterable['Segment']:
    """Remove all links from an iterable of styles.

        Args:
            segments (Iterable[Segment]): An iterable segments.

        Yields:
            Segment: Segments with link removed.
        """
    for segment in segments:
        if segment.control or segment.style is None:
            yield segment
        else:
            text, style, _control = segment
            yield cls(text, style.update_link(None) if style else None)