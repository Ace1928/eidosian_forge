from __future__ import annotations
import warnings
from typing import Optional
from attrs import define, field
from fontTools.misc.transform import Identity, Transform
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import AbstractPointPen, PointToSegmentPen
from ufoLib2.objects.misc import BoundingBox
from ufoLib2.serde import serde
from ufoLib2.typing import GlyphSet
from .misc import _convert_transform, getBounds, getControlBounds
def getControlBounds(self, layer: GlyphSet) -> BoundingBox | None:
    """Returns the (xMin, yMin, xMax, yMax) bounding box of the component,
        taking only the control points into account.

        Args:
            layer: The layer of the containing glyph to look up components.
        """
    return getControlBounds(self, layer)