from collections import namedtuple
import math
import warnings
def cos_sin_deg(deg: float):
    """Return the cosine and sin for the given angle in degrees.

    With special-case handling of multiples of 90 for perfect right
    angles.
    """
    deg = deg % 360.0
    if deg == 90.0:
        return (0.0, 1.0)
    elif deg == 180.0:
        return (-1.0, 0)
    elif deg == 270.0:
        return (0, -1.0)
    rad = math.radians(deg)
    return (math.cos(rad), math.sin(rad))