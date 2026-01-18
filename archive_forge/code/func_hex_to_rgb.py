import math
import warnings
import matplotlib.dates
def hex_to_rgb(value):
    """
    Change a hex color to an rgb tuple

    :param (str|unicode) value: The hex string we want to convert.
    :return: (int, int, int) The red, green, blue int-tuple.

    Example:

        '#FFFFFF' --> (255, 255, 255)

    """
    value = value.lstrip('#')
    lv = len(value)
    return tuple((int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)))