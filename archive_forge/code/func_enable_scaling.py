from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Protocol, List
import pyglet
from pyglet.font import base
def enable_scaling(self, base_size: int) -> None:
    super().enable_scaling(base_size)
    glyphs = self.get_glyphs(self.default_char)
    self.ascent = glyphs[0].height
    self.descent = 0