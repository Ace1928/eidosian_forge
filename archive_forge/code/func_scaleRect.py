from fontTools.misc.roundTools import otRound
from fontTools.misc.vector import Vector as _Vector
import math
import warnings
def scaleRect(rect, x, y):
    """Scale a bounding box rectangle.

    Args:
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.
        x: Factor to scale the rectangle along the X axis.
        Y: Factor to scale the rectangle along the Y axis.

    Returns:
        A scaled bounding rectangle.
    """
    xMin, yMin, xMax, yMax = rect
    return (xMin * x, yMin * y, xMax * x, yMax * y)