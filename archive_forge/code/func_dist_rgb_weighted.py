from math import cos, exp, sin, sqrt, atan2
def dist_rgb_weighted(rgb1, rgb2):
    """
    Determine the weighted distance between two rgb colors.

    :arg tuple rgb1: RGB color definition
    :arg tuple rgb2: RGB color definition
    :returns: Square of the distance between provided colors
    :rtype: float

    Similar to a standard distance formula, the values are weighted
    to approximate human perception of color differences

    For efficiency, the square of the distance is returned
    which is sufficient for comparisons
    """
    red_mean = (rgb1[0] + rgb2[0]) / 2.0
    return (2 + red_mean / 256) * pow(rgb1[0] - rgb2[0], 2) + 4 * pow(rgb1[1] - rgb2[1], 2) + (2 + (255 - red_mean) / 256) * pow(rgb1[2] - rgb2[2], 2)