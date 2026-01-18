from fontTools.misc.roundTools import otRound
from fontTools.misc.vector import Vector as _Vector
import math
import warnings
def rectCenter(rect):
    """Determine rectangle center.

    Args:
        rect: Bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.

    Returns:
        A 2D tuple representing the point at the center of the rectangle.
    """
    xMin, yMin, xMax, yMax = rect
    return ((xMin + xMax) / 2, (yMin + yMax) / 2)