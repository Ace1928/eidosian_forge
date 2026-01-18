from collections import namedtuple
import math
import warnings
def loadsw(s: str):
    """Returns Affine from the contents of a world file string.

    This method also translates the coefficients from center- to
    corner-based coordinates.

    :param s: str with 6 floats ordered in a world file.
    :rtype: Affine
    """
    if not hasattr(s, 'split'):
        raise TypeError('Cannot split input string')
    coeffs = s.split()
    if len(coeffs) != 6:
        raise ValueError('Expected 6 coefficients, found %d' % len(coeffs))
    a, d, b, e, c, f = (float(x) for x in coeffs)
    center = tuple.__new__(Affine, [a, b, c, d, e, f, 0.0, 0.0, 1.0])
    return center * Affine.translation(-0.5, -0.5)