from fontTools.misc.roundTools import otRound
from fontTools.misc.vector import Vector as _Vector
import math
import warnings
def pointsInRect(array, rect):
    """Determine which points are inside a bounding rectangle.

    Args:
        array: A sequence of 2D tuples.
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.

    Returns:
        A list containing the points inside the rectangle.
    """
    if len(array) < 1:
        return []
    xMin, yMin, xMax, yMax = rect
    return [xMin <= x <= xMax and yMin <= y <= yMax for x, y in array]