from fontTools.misc.roundTools import otRound
from fontTools.misc.vector import Vector as _Vector
import math
import warnings
def sectRect(rect1, rect2):
    """Test for rectangle-rectangle intersection.

    Args:
        rect1: First bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.
        rect2: Second bounding rectangle.

    Returns:
        A boolean and a rectangle.
        If the input rectangles intersect, returns ``True`` and the intersecting
        rectangle. Returns ``False`` and ``(0, 0, 0, 0)`` if the input
        rectangles don't intersect.
    """
    xMin1, yMin1, xMax1, yMax1 = rect1
    xMin2, yMin2, xMax2, yMax2 = rect2
    xMin, yMin, xMax, yMax = (max(xMin1, xMin2), max(yMin1, yMin2), min(xMax1, xMax2), min(yMax1, yMax2))
    if xMin >= xMax or yMin >= yMax:
        return (False, (0, 0, 0, 0))
    return (True, (xMin, yMin, xMax, yMax))