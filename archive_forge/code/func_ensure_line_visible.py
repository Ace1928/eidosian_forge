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
def ensure_line_visible(self, line_idx: int) -> None:
    """Adjust `view_y` so that the line with the given index is visible.

        :Parameters:
            `line` : int
                Line index.

        """
    line = self.lines[line_idx]
    y1 = line.y + line.ascent
    y2 = line.y + line.descent
    if y1 > self.view_y:
        self.view_y = y1
    elif y2 < self.view_y - self.height:
        self.view_y = y2 + self.height
    elif abs(self.view_y) > self.content_height - self.height:
        self.view_y = -self.content_height