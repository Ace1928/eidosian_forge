from __future__ import annotations
from typing import (
from attrs import define, field
from fontTools.ufoLib.glifLib import GlyphSet
from ufoLib2.constants import DEFAULT_LAYER_NAME
from ufoLib2.objects.glyph import Glyph
from ufoLib2.objects.lib import (
from ufoLib2.objects.misc import (
from ufoLib2.serde import serde
from ufoLib2.typing import T
def newGlyph(self, name: str) -> Glyph:
    """Creates and returns new Glyph object in this layer with name."""
    if name in self._glyphs:
        raise KeyError(f"glyph named '{name}' already exists")
    self._glyphs[name] = glyph = Glyph(name)
    return glyph