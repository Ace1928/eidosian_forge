from collections import namedtuple
import math
import warnings
def dumpsw(obj) -> str:
    """Return string for a world file.

    This method also translates the coefficients from corner- to
    center-based coordinates.

    :rtype: str
    """
    center = obj * Affine.translation(0.5, 0.5)
    return '\n'.join((repr(getattr(center, x)) for x in list('adbecf'))) + '\n'