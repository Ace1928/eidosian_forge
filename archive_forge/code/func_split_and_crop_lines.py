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
def split_and_crop_lines(cls, segments: Iterable['Segment'], length: int, style: Optional[Style]=None, pad: bool=True, include_new_lines: bool=True) -> Iterable[List['Segment']]:
    """Split segments in to lines, and crop lines greater than a given length.

        Args:
            segments (Iterable[Segment]): An iterable of segments, probably
                generated from console.render.
            length (int): Desired line length.
            style (Style, optional): Style to use for any padding.
            pad (bool): Enable padding of lines that are less than `length`.

        Returns:
            Iterable[List[Segment]]: An iterable of lines of segments.
        """
    line: List[Segment] = []
    append = line.append
    adjust_line_length = cls.adjust_line_length
    new_line_segment = cls('\n')
    for segment in segments:
        if '\n' in segment.text and (not segment.control):
            text, segment_style, _ = segment
            while text:
                _text, new_line, text = text.partition('\n')
                if _text:
                    append(cls(_text, segment_style))
                if new_line:
                    cropped_line = adjust_line_length(line, length, style=style, pad=pad)
                    if include_new_lines:
                        cropped_line.append(new_line_segment)
                    yield cropped_line
                    line.clear()
        else:
            append(segment)
    if line:
        yield adjust_line_length(line, length, style=style, pad=pad)