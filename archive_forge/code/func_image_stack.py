from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
@glyph_method(glyphs.ImageStack)
def image_stack(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
    pass