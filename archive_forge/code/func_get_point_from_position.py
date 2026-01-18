from __future__ import annotations
import sys
from typing import List, Optional, Type, Any, Tuple, TYPE_CHECKING
from pyglet.customtypes import AnchorX, AnchorY
from pyglet.event import EventDispatcher
from pyglet.font.base import grapheme_break
from pyglet.text import runlist
from pyglet.text.document import AbstractDocument
from pyglet.text.layout.base import _is_pyglet_doc_run, _Line, _LayoutContext, _InlineElementBox, _InvalidRange, \
from pyglet.text.layout.scrolling import ScrollableTextLayoutGroup, ScrollableTextDecorationGroup
def get_point_from_position(self, position, line=None):
    """Get the X, Y coordinates of a position in the document.

        The position that ends a line has an ambiguous point: it can be either
        the end of the line, or the beginning of the next line.  You may
        optionally specify a line index to disambiguate the case.

        The resulting Y coordinate gives the baseline of the line.

        :Parameters:
            `position` : int
                Character position within document.
            `line` : int
                Line index.

        :rtype: (int, int)
        :return: (x, y)
        """
    if line is None:
        line = self.lines[0]
        for next_line in self.lines:
            if next_line.start > position:
                break
            line = next_line
    else:
        line = self.lines[line]
    x = line.x
    baseline = self._document.get_style('baseline', max(0, position - 1))
    if baseline is None:
        baseline = 0
    else:
        baseline = self.parse_distance(baseline)
    position -= line.start
    for box in line.boxes:
        if position - box.length <= 0:
            x += box.get_point_in_box(position)
            break
        position -= box.length
        x += box.advance
    return (x, line.y + baseline)