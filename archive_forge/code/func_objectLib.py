from __future__ import annotations
from copy import deepcopy
from typing import Any, Iterator, List, Mapping, Optional, cast
from attrs import define, field
from fontTools.misc.transform import Transform
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import (
from ufoLib2.objects.anchor import Anchor
from ufoLib2.objects.component import Component
from ufoLib2.objects.contour import Contour
from ufoLib2.objects.guideline import Guideline
from ufoLib2.objects.image import Image
from ufoLib2.objects.lib import (
from ufoLib2.objects.misc import BoundingBox, _object_lib, getBounds, getControlBounds
from ufoLib2.pointPens.glyphPointPen import GlyphPointPen
from ufoLib2.serde import serde
from ufoLib2.typing import GlyphSet, HasIdentifier
def objectLib(self, object: HasIdentifier) -> dict[str, Any]:
    """Return the lib for an object with an identifier, as stored in a glyph's lib.

        If the object does not yet have an identifier, a new one is assigned to it. If
        the font lib does not yet contain the object's lib, a new one is inserted and
        returned.

        .. doctest::

            >>> from ufoLib2.objects import Font, Guideline
            >>> font = Font()
            >>> glyph = font.newGlyph("a")
            >>> glyph.guidelines = [Guideline(x=100)]
            >>> guideline_lib = glyph.objectLib(glyph.guidelines[0])
            >>> guideline_lib["com.test.foo"] = 1234
            >>> guideline_id = glyph.guidelines[0].identifier
            >>> assert guideline_id is not None
            >>> assert glyph.lib["public.objectLibs"][guideline_id] is guideline_lib
        """
    return _object_lib(self.lib, object)