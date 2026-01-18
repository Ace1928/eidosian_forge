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
def insertGlyph(self, glyph: Glyph, name: str | None=None, overwrite: bool=True, copy: bool=True) -> None:
    """Inserts Glyph object into this layer.

        Args:
            glyph: The Glyph object.
            name: The name of the glyph.
            overwrite: If True, overwrites (read: deletes) glyph with the same name if
                it exists. If False, raises KeyError.
            copy: If True, copies the Glyph object before insertion. If False, inserts
                as is.
        """
    if copy:
        glyph = glyph.copy()
    if name is not None:
        glyph._name = name
    if glyph.name is None:
        raise ValueError(f"{glyph!r} has no name; can't add it to Layer")
    if not overwrite and glyph.name in self._glyphs:
        raise KeyError(f"glyph named '{glyph.name}' already exists")
    self._glyphs[glyph.name] = glyph