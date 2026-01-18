from math import cos, exp, sin, sqrt, atan2
@lru_cache(maxsize=256)
def rgb_to_lab(red, green, blue):
    """
    Convert RGB color to CIE-Lab color.

    :arg int red: RGB value of Red.
    :arg int green: RGB value of Green.
    :arg int blue: RGB value of Blue.
    :returns: Tuple (L, a, b) representing CIE-Lab color
    :rtype: tuple

    D65/2° standard illuminant
    """
    return xyz_to_lab(*rgb_to_xyz(red, green, blue))