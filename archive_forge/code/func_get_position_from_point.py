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
def get_position_from_point(self, x: float, y: float) -> int:
    """Get the closest document position to a point.

        :Parameters:
            `x` : int
                X coordinate
            `y` : int
                Y coordinate

        """
    line = self.get_line_from_point(x, y)
    return self.get_position_on_line(line, x)