from fontTools.misc.roundTools import otRound
from fontTools.misc.vector import Vector as _Vector
import math
import warnings
def quantizeRect(rect, factor=1):
    """
    >>> bounds = (72.3, -218.4, 1201.3, 919.1)
    >>> quantizeRect(bounds)
    (72, -219, 1202, 920)
    >>> quantizeRect(bounds, factor=10)
    (70, -220, 1210, 920)
    >>> quantizeRect(bounds, factor=100)
    (0, -300, 1300, 1000)
    """
    if factor < 1:
        raise ValueError(f'Expected quantization factor >= 1, found: {factor!r}')
    xMin, yMin, xMax, yMax = normRect(rect)
    return (int(math.floor(xMin / factor) * factor), int(math.floor(yMin / factor) * factor), int(math.ceil(xMax / factor) * factor), int(math.ceil(yMax / factor) * factor))