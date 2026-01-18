from array import array
from typing import Any, Callable, Dict, Optional, Tuple
from fontTools.misc.fixedTools import MAX_F2DOT14, floatToFixedToFloat
from fontTools.misc.loggingTools import LogMixin
from fontTools.pens.pointPen import AbstractPointPen
from fontTools.misc.roundTools import otRound
from fontTools.pens.basePen import LoggingPen, PenError
from fontTools.pens.transformPen import TransformPen, TransformPointPen
from fontTools.ttLib.tables import ttProgram
from fontTools.ttLib.tables._g_l_y_f import flagOnCurve, flagCubic
from fontTools.ttLib.tables._g_l_y_f import Glyph
from fontTools.ttLib.tables._g_l_y_f import GlyphComponent
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
from fontTools.ttLib.tables._g_l_y_f import dropImpliedOnCurvePoints
import math
def glyph(self, componentFlags: int=4, dropImpliedOnCurves: bool=False, *, round: Callable[[float], int]=otRound) -> Glyph:
    """
        Returns a :py:class:`~._g_l_y_f.Glyph` object representing the glyph.

        Args:
            componentFlags: Flags to use for component glyphs. (default: 0x04)

            dropImpliedOnCurves: Whether to remove implied-oncurve points. (default: False)
        """
    if not self._isClosed():
        raise PenError("Didn't close last contour.")
    components = self._buildComponents(componentFlags)
    glyph = Glyph()
    glyph.coordinates = GlyphCoordinates(self.points)
    glyph.endPtsOfContours = self.endPts
    glyph.flags = array('B', self.types)
    self.init()
    if components:
        glyph.components = components
        glyph.numberOfContours = -1
    else:
        glyph.numberOfContours = len(glyph.endPtsOfContours)
        glyph.program = ttProgram.Program()
        glyph.program.fromBytecode(b'')
        if dropImpliedOnCurves:
            dropImpliedOnCurvePoints(glyph)
        glyph.coordinates.toInt(round=round)
    return glyph