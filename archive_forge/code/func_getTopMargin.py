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
def getTopMargin(self, layer: GlyphSet | None=None) -> float | None:
    """Returns the the space in font units from the top of the canvas to
        the top of the glyph.

        Args:
            layer: The layer of the glyph to look up components, if any. Not needed for
                pure-contour glyphs.
        """
    bounds = self.getBounds(layer)
    if bounds is None:
        return None
    if self.verticalOrigin is None:
        return self.height - bounds.yMax
    else:
        return self.verticalOrigin - bounds.yMax