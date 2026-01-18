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
def on_delete_text(self, start: int, end: int) -> None:
    self.glyphs[start:end] = []
    if self._multiline and self._content_valign != 'top':
        visible_line = self.lines[self.visible_lines.start]
        self.invalid_flow.invalidate(visible_line.start, end)
    self.invalid_glyphs.delete(start, end)
    self.invalid_flow.delete(start, end)
    self.invalid_style.delete(start, end)
    self.owner_runs.delete(start, end)
    size = end - start
    for line in self.lines:
        if line.start > start:
            line.start = max(line.start - size, start)
    if start == 0:
        self.invalid_flow.invalidate(0, 1)
    else:
        self.invalid_flow.invalidate(start - 1, start)
    self._update()