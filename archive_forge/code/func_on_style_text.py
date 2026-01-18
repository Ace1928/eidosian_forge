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
def on_style_text(self, start: int, end: int, attributes: dict[str, Any]) -> None:
    if 'font_name' in attributes or 'font_size' in attributes or 'bold' in attributes or ('italic' in attributes):
        self.invalid_glyphs.invalidate(start, end)
    elif 'color' in attributes or 'background_color' in attributes:
        self.invalid_style.invalidate(start, end)
    else:
        self.invalid_flow.invalidate(start, end)
    self._update()