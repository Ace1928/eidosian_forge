from fontTools.misc.roundTools import otRound
from fontTools.misc.vector import Vector as _Vector
import math
import warnings
def rectArea(rect):
    """Determine rectangle area.

    Args:
        rect: Bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.

    Returns:
        The area of the rectangle.
    """
    xMin, yMin, xMax, yMax = rect
    return (yMax - yMin) * (xMax - xMin)