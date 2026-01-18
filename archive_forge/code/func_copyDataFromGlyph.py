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
def copyDataFromGlyph(self, glyph: Glyph) -> None:
    """Deep-copies everything from the other glyph into self, except for
        the name.

        Existing glyph data is overwritten.

        |defcon_compat|
        """
    self.width = glyph.width
    self.height = glyph.height
    self.unicodes = list(glyph.unicodes)
    self.image = deepcopy(glyph.image)
    self.note = glyph.note
    self.lib = deepcopy(glyph.lib)
    self.anchors = deepcopy(glyph.anchors)
    self.guidelines = deepcopy(glyph.guidelines)
    self.clearContours()
    self.clearComponents()
    pointPen = self.getPointPen()
    glyph.drawPoints(pointPen)