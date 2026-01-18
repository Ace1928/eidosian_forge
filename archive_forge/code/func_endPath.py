from __future__ import annotations
from typing import TYPE_CHECKING, Any
from fontTools.misc.transform import Transform
from fontTools.pens.pointPen import AbstractPointPen
from ufoLib2.objects.component import Component
from ufoLib2.objects.contour import Contour
from ufoLib2.objects.point import Point
def endPath(self) -> None:
    if self._contour is None:
        raise ValueError('Call beginPath first.')
    self._glyph.contours.append(self._contour)
    self._contour = None