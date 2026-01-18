from math import cos, exp, sin, sqrt, atan2
def dist_cie76(rgb1, rgb2):
    """
    Determine distance between two rgb colors using the CIE94 algorithm.

    :arg tuple rgb1: RGB color definition
    :arg tuple rgb2: RGB color definition
    :returns: Square of the distance between provided colors
    :rtype: float

    For efficiency, the square of the distance is returned
    which is sufficient for comparisons
    """
    l_1, a_1, b_1 = rgb_to_lab(*rgb1)
    l_2, a_2, b_2 = rgb_to_lab(*rgb2)
    return pow(l_1 - l_2, 2) + pow(a_1 - a_2, 2) + pow(b_1 - b_2, 2)