from fontTools.misc.roundTools import otRound
from fontTools.misc.vector import Vector as _Vector
import math
import warnings
def asInt16(array):
    """Round a list of floats to 16-bit signed integers.

    Args:
        array: List of float values.

    Returns:
        A list of rounded integers.
    """
    return [int(math.floor(i + 0.5)) for i in array]